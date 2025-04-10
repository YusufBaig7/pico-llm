# starter code by matus & o1-pro
import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# We do not import numpy or scikit-learn, so we implement a naive k-means in pure PyTorch.
# If you prefer scikit-learn, you can adapt the code.

from datasets import load_dataset
import tiktoken

################################################################################
# 1. Command-line arg parsing
################################################################################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files."
    )
    parser.add_argument(
        "--input_files",
        nargs="*",
        default=None,
        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).",
    )
    parser.add_argument(
        "--tinystories_weight",
        type=float,
        default=0.5,
        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).",
    )
    parser.add_argument(
        "--max_steps_per_epoch",
        type=int,
        default=None,
        help="If set, each epoch ends after this many steps (for quick tests).",
    )
    parser.add_argument(
        "--num_inner_mlp_layers",
        type=int,
        default=1,
        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.",
    )
    parser.add_argument(
        "--monosemantic_enabled",
        action="store_true",
        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.",
    )
    parser.set_defaults(monosemantic_enabled=True)  # disable by default

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument(
        "--kgram_k",
        type=int,
        default=3,
        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.",
    )
    parser.add_argument(
        "--kgram_chunk_size",
        type=int,
        default=1,
        help="Process k-gram timesteps in micro-batches. Default=1.",
    )

    parser.add_argument(
        "--block_size",
        type=int,
        default=1024,
        help="Maximum sequence length for each example. Default=1024.",
    )

    # New arguments:
    parser.add_argument(
        "--embed_size",
        type=int,
        default=1024,
        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=1024.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a",
        help="Prompt used for generation. Default='Once upon a'.",
    )

    # Newly added device argument:
    parser.add_argument(
        "--device_id",
        type=str,
        default="cuda:0",
        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.",
    )

    args = parser.parse_args()
    return args


################################################################################
# 2. Data handling: entire sequences up to block_size => (seq_len, batch)
################################################################################


class MixedSequenceDataset(torch.utils.data.Dataset):
    """
    We store two lists of entire token sequences:
      - tinystories_seqs
      - other_seqs
    Each sequence is length <= block_size.

    During __getitem__, we randomly pick from one list or the other with probability p_tiny.
    Return that entire sequence as a 1D LongTensor.
    """

    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny

        self.has_tinystories = len(self.tinystories_seqs) > 0
        self.has_other = len(self.other_seqs) > 0

        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError(
                "No data found! Both TinyStories and other sets are empty."
            )

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        return torch.tensor(seq, dtype=torch.long)


def seq_collate_fn(batch):
    """
    batch: list of 1D LongTensors of various lengths [<= block_size].
    1) find max length
    2) pad with zeros
    3) shape => (max_len, batch_size)
    """
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    return padded


################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach
################################################################################


def compute_next_token_loss(logits, tokens):
    """
    logits: (seq_len, batch, vocab_size)
    tokens: (seq_len, batch)
    Next-token prediction => we shift target by 1.
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
    gold = tokens[1:, :]  # (seq_len-1, batch)

    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)
    return F.cross_entropy(preds, gold)


class KGramMLPSeqModel(nn.Module):
    """
    For each position t in [0..seq_len-1], gather the last k tokens => one-hot => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Potentially very large memory usage for big vocab or seq_len. chunk_size helps mitigate overhead.
    """

    def __init__(
        self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1
    ):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        input_size = k * vocab_size
        layers = []
        current_size = input_size

        for _ in range(num_inner_layers):
            layers.append(nn.Linear(current_size, embed_size))
            layers.append(nn.SiLU())
            current_size = embed_size

        layers.append(nn.Linear(current_size, vocab_size))

        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq):
        seq_len, batch_size = tokens_seq.shape
        k = self.k  # context window size
        # Step 1: Pre-pad tokens_seq with zeros for the initial context (shape: (k, batch))
        zeros = torch.zeros(
                    (k, batch_size), dtype=tokens_seq.dtype, device=tokens_seq.device
                )
        padded = torch.cat([zeros, tokens_seq], dim=0)  # (seq_len + k, batch)
        # Step 2: Create a sliding window view for each batch element.
        # Transpose so that batch is the first dimension: (batch, seq_len + k)
        padded = padded.transpose(0, 1)
        # Use unfold to extract sliding windows of size k along the time dimension.
        # This gives shape: (batch, seq_len + 1, k) because:
        #  (seq_len + k) - k + 1 = seq_len + 1.
        contexts = padded.unfold(dimension=1, size=k, step=1)
        # We only need seq_len windows, so slice to drop the extra one.
        contexts = contexts[:, :seq_len, :]  # (batch, seq_len, k)
        # Rearrange to (seq_len, batch, k)
        contexts = contexts.transpose(0, 1)
        # Step 3: Vectorized one-hot encoding and flattening the context window.
        # One-hot encode: shape becomes (seq_len, batch, k, vocab_size)
        contexts_oh = F.one_hot(contexts, num_classes=self.vocab_size).float()
        # Flatten last two dimensions -> (seq_len, batch, k * vocab_size)
        contexts_flat = contexts_oh.view(seq_len, batch_size, -1)
        # Step 4: Process all contexts in one network call.
        # Reshape to process all contexts at once (shape: (seq_len * batch, k * vocab_size))
        contexts_flat = contexts_flat.view(seq_len * batch_size, -1)
        logits = self.net(contexts_flat)  # (seq_len * batch, vocab_size)
        # Reshape the output back to the expected dimensions: (seq_len, batch, vocab_size)
        outputs = logits.view(seq_len, batch_size, self.vocab_size)
        return outputs


################################################################################
# 4. LSTM-based seq2seq
################################################################################


class LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        => (seq_len, batch, vocab_size)
        """
        emb = self.embedding(tokens_seq)  # (seq_len, batch, embed)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)  # (seq_len, batch, hidden)
        logits = self.linear(out)  # (seq_len, batch, vocab_size)
        return logits


################################################################################
# 5. Our "stub" Transformer with KV-cache
#    Very slow Python loop for training. Multi-head sums head outputs.
################################################################################

activations = {}

def save_activation(name):
    def hook(module, input, output):
        print("Hello00000000000000")
        # Save the output activation (detach to avoid tracking gradients)
        activations[name] = output.detach().cpu()
    return hook


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / norm)
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(RotaryPositionalEmbedding, self).__init__()

        self.rotation_matrix = torch.zeros(d_model, d_model, device=torch.device("cuda"))
        for i in range(d_model):
            for j in range(d_model):
                self.rotation_matrix[i, j] = math.cos(i * j * 0.01)

        self.positional_embedding = torch.zeros(max_seq_len, d_model, device=torch.device("cuda"))
        for i in range(max_seq_len):
            for j in range(d_model):
                self.positional_embedding[i, j] = math.cos(i * j * 0.01)

    def forward(self, x):

        x += self.positional_embedding[:x.size(1), :]
        x = torch.matmul(x, self.rotation_matrix)
        return x

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):

        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        pos_encoding = self.pe[:, :seq_len]
        return x + pos_encoding

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, attn_dropout = 0.1, resid_dropout = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.scale = self.head_size ** -0.5

        self.qkv_proj = nn.Linear(d_model, 3*d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim =2)

        q = q.reshape(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = k.reshape(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.head_size).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale

        att = attn.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
        att = torch.softmax(att, dim = -1)

        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y
                
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size, attn_dropout = 0.1, resid_dropout = 0.1, mlp_ratio = 4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, attn_dropout, resid_dropout)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(resid_dropout),
            )
            
    def forward(self, x):
        x = x + self.attn(self.ln1(x))

        x = x + self.mlp(self.ln2(x))
        return x
        

class TransformerModel(nn.Module):
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=2, n_blocks=4, block_size = 1024, attn_dropout = 0.1, resid_dropout = 0.1, use_rope = False, use_nope = False, use_sinu = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        if use_nope:
            self.pos_embedding = None
        elif use_rope:
            self.pos_embedding = RotaryPositionalEmbedding(d_model, block_size)
        elif use_sinu:
            self.pos_embedding = SinusoidalPositionalEmbedding(d_model, block_size)
        else:
            self.pos_embedding = nn.Embedding(block_size, d_model)
        
        self.emb_dropout = nn.Dropout(resid_dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, block_size, attn_dropout, resid_dropout, mlp_ratio= 0.4)
            for _ in range(n_blocks)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, tokens_seq):

        tokens_seq = tokens_seq.transpose(0, 1)
        B, T = tokens_seq.shape
        
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"
        #tok_emb = self.token_embedding(tokens_seq) * math.sqrt(self.d_model)    sinusoidal
        # tok_emb = self.token_embedding(tokens_seq) 
        
        # x = self.pos_embedding(tok_emb) # Absolute Embeddings
        # positions = torch.arange(0, T, dtype=torch.long, device=tokens_seq.device)  # Absolute Embeddings
        # pos_emb = self.pos_embedding(positions).unsqueeze(0)  # Absolute Embeddings
        # x = tok_emb + pos_emb  # Absolute Embeddings

        # if self.pos_embedding is not None:
        #     tok_emb = self.pos_embedding(tok_emb)

        tok_emb = self.token_embedding(tokens_seq)
        positions = torch.arange(0, T, dtype=torch.long, device=tokens_seq.device)
        pos_emb = self.pos_embedding(positions).unsqueeze(0)
        
        x = tok_emb + pos_emb
        x = self.emb_dropout(x)
        
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits.transpose(0, 1)
    


        

        



################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################


def monosemantic_analysis_for_token(token_id, model, enc, device="cpu", top_n=5):
    """
    Perform a simple analysis to rank neurons by their consistency (low variance)
    for a given token. This function assumes that a forward pass has been run such
    that the global 'activations' dict contains an entry for 'last_mlp'.
    
    Args:
        token_id (int): The token ID for which to perform analysis.
        model (nn.Module): The transformer model.
        enc: The tokenizer/encoding object.
        device (str): Device used for analysis.
        top_n (int): Number of top neurons to return.
        
    Returns:
        List of tuples (neuron_index, variance) for the top_n neurons with lowest variance.
    """
    # Ensure the model runs in evaluation mode.
    model.eval()

    # Prepare a prompt that includes the token of interest.
    # For demonstration, we assume that 'token_id' appears at the last position.
    # You might wish to craft inputs more carefully in practice.
    token_str = enc.decode([token_id])
    prompt = f"This is a sentence that ends with the token: {token_str}"
    context_tokens = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(1)
    
    # Run the model forward (activations will be captured by the hook).
    with torch.no_grad():
        _ = model(context_tokens)

    print("Activations:", activations)
    
    # Retrieve the activations from the last MLP layer.
    # Expected shape: (batch, seq_len, hidden_size)
    if "last_mlp" not in activations:
        print("No activations recorded. Ensure the hook is registered correctly.")
        return []
    
    act = activations["last_mlp"]
    
    # For analysis, we focus on the token position where our token of interest appears.
    # Here we assume it is the last token in the sequence.
    token_activation = act[:, -1, :]  # shape: (batch, hidden_size)
    
    # Compute the variance for each neuron (across batch samples).
    # Lower variance implies more consistent (monosemantic) behavior.
    variances = token_activation.var(dim=0)  # shape: (hidden_size,)
    
    # Identify the top_n neurons with the smallest variance.
    # (We use negative variance for topk to pick the smallest values.)
    top_vars, top_indices = torch.topk(-variances, top_n)
    
    # Prepare and return the results as a list of tuples (neuron_index, variance).
    results = [(idx.item(), variances[idx].item()) for idx in top_indices]
    return results



################################################################################
# 7. Single code path for text generation
################################################################################


def nucleus_sampling(logits, p=0.95):
    """
    Implements nucleus (top-p) sampling for text generation.
    
    Args:
        logits: Raw model output logits of shape (vocab_size,)
        p: Probability threshold (default: 0.95)
        
    Returns:
        Integer token ID sampled from the truncated distribution
    """
    # Convert logits to probabilities using softmax
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus_mask = cumulative_probs <= p

    if not nucleus_mask.any():
        return torch.argmax(logits).item()
    
    nucleus_probs = sorted_probs[nucleus_mask]
    nucleus_indices = sorted_indices[nucleus_mask]

    nucleus_probs = nucleus_probs / nucleus_probs.sum()

    sample_idx = torch.multinomial(nucleus_probs, num_samples=1).item()
    
    return nucleus_indices[sample_idx].item()

def generate_text(
    model,
    enc,
    init_text,
    max_new_tokens=20,
    device="cpu",
    top_p=None,
    monosemantic_info=None,
    do_monosemantic=True,
):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step, we feed the entire context as (seq_len,1) to model(...).
      - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
      - We pick next token (greedy or top-p), append to context_tokens.
      - Optionally do monosemantic analysis on that newly generated token.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        annotation_list = []

        for step_i in range(max_new_tokens):
            seq_tensor = torch.tensor(
                context_tokens, dtype=torch.long, device=device
            ).unsqueeze(1)
            logits_seq = model(seq_tensor)  # (seq_len,1,vocab_size)
            next_logits = logits_seq[-1, 0, :]  # shape (vocab_size,)

            if top_p is None:
                # greedy
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(next_logits, p=top_p)

            context_tokens.append(chosen_token)

            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    model.train(was_training)

    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    for tid, neighs in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            print("------------------------------------Neighbour Stats")
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text


################################################################################
# 8. Training
################################################################################


def train_one_model(
    model,
    loader,
    epochs,
    model_name,
    device,
    lr=1e-3,
    log_steps=100,
    sample_interval=30,
    max_steps_per_epoch=None,
    enc=None,
    monosemantic_info=None,
    prompt="Once upon a",
):
    """
    We add `prompt` as an explicit argument so we can pass it down from main().
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    next_sample_time = start_time
    global_step = 0
    loss_history = []            # All training losses (per batch)
    log_global_steps = []        # Global steps for which logging occurs
    log_partial_losses = [] 

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0

        step_in_epoch = 0
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1

            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                print(
                    f"[{model_name}] Epoch {epoch}/{epochs}, "
                    f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                    f"Partial Avg Loss: {avg_part_loss:.4f}"
                )
                log_global_steps.append(global_step)
                log_partial_losses.append(avg_part_loss)
                partial_loss = 0.0
                partial_count = 0

            current_time = time.time()
            if current_time >= next_sample_time and enc is not None:
                with torch.no_grad():
                    print(
                        f"\n[{model_name}] Generating sample text (greedy) at epoch={epoch}, step={batch_idx}..."
                    )
                    text_greedy, ann_greedy = generate_text(
                        model,
                        enc,
                        prompt,
                        max_new_tokens=20,
                        device=device,
                        top_p=None,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None),
                    )
                    print(f" Greedy Sample: {text_greedy}")
                    print(f" Annotated: {ann_greedy}\n")

                    print(
                        f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}..."
                    )
                    text_topp, ann_topp = generate_text(
                        model,
                        enc,
                        prompt,
                        max_new_tokens=20,
                        device=device,
                        top_p=0.95,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None),
                    )
                    print(f" Top-p (p=0.95) Sample: {text_topp}")
                    print(f" Annotated: {ann_topp}\n")

                    # third generation => top-p=1.0 => full distribution random sampling
                    print(
                        f"[{model_name}] Generating sample text (top-p=1.0) at epoch={epoch}, step={batch_idx}..."
                    )
                    text_topp1, ann_topp1 = generate_text(
                        model,
                        enc,
                        prompt,
                        max_new_tokens=20,
                        device=device,
                        top_p=1.0,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None),
                    )
                    print(f" Top-p (p=1.0) Sample: {text_topp1}")
                    print(f" Annotated: {ann_topp1}\n")

                next_sample_time = current_time + sample_interval

            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(
                    f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early."
                )
                break

        avg_loss = total_loss / step_in_epoch
        print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Loss: {avg_loss:.4f}")


    plt.figure(figsize=(10, 6))
    plt.plot(log_global_steps, log_partial_losses, marker='o', linestyle='-', label='Partial Avg Loss')
    plt.xlabel('Global Step')
    plt.ylabel('Partial Average Loss')
    plt.title('Partial Avg Loss vs Global Steps')
    plt.legend()
    plt.grid(True)
    plt.show()



################################################################################
# 9. Main
################################################################################


def main():
    args = parse_args()

    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    embed_size = args.embed_size
    batch_size = 16
    num_epochs = 3
    learning_rate = 1e-3

    block_size = args.block_size
    train_subset_size = 20000
    log_interval_steps = 100
    sample_interval_seconds = 30

    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers

    # NEW: pick device from args.device_id, fallback to cpu if needed
    requested_device_id = args.device_id
    if requested_device_id.startswith("mps") and not torch.backends.mps.is_available():
        print(
            f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU."
        )
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(
        f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size}"
    )

    ############################################################################
    # Data
    ############################################################################
    tinystories_seqs = []
    other_seqs = []

    if args.tinystories_weight > 0.0:
        print(
            f"Loading TinyStories from huggingface with weight={args.tinystories_weight}..."
        )
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    if dataset is not None:
        for sample in dataset:
            text = sample["text"]
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")

    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")

    p_tiny = args.tinystories_weight
    if len(tinystories_seqs) == 0 and p_tiny > 0:
        print(
            "Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it."
        )
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs, other_seqs=other_seqs, p_tiny=p_tiny
    )

    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn,
    )

    ############################################################################
    # Models
    ############################################################################
    kgram_model = KGramMLPSeqModel(
        vocab_size=vocab_size,
        k=k,
        embed_size=embed_size,
        num_inner_layers=num_inner_layers,
        chunk_size=chunk_size,
    ).to(device)

    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size, embed_size=embed_size, hidden_size=embed_size
    ).to(device)

    transformer = TransformerModel(
    vocab_size=vocab_size,
    d_model=512,
    n_heads=8,
    n_blocks=6,
    block_size=block_size,  # matches the --block_size argument
    attn_dropout=0.1,
    resid_dropout=0.1,
    use_rope= False,
    use_nope= False,
    use_sinu= False,
    ).to(device)

    #transformer = TransformerModel().to(device)

    models = {
        #"kgram_mlp_seq": kgram_model,
        #"lstm_seq": lstm_model,
        "transformer_seq": transformer,
    }

    ############################################################################
    # Train each model
    ############################################################################
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        train_one_model(
            model=model,
            loader=train_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt,  # <--- Pass the user-specified prompt here
        )

        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model,
                enc,
                args.prompt,
                max_new_tokens=50,
                device=device,
                top_p=None,
            )
            # 2) top-p=0.95
            text_topp, ann_topp = generate_text(
                model,
                enc,
                args.prompt,
                max_new_tokens=50,
                device=device,
                top_p=0.95,
            )
            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model,
                enc,
                args.prompt,
                max_new_tokens=50,
                device=device,
                top_p=1.0,
            )

        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)
        print(f"Annotated:\n{ann_greedy}\n")

        print(f"[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp)
        print(f"Annotated:\n{ann_topp}\n")

        print(f"[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)
        print(f"Annotated:\n{ann_topp1}")
        print("--------------------------------------------------")

    # Finally, let's share how I'm feeling:


    if args.monosemantic_enabled:
        transformer.blocks[-1].mlp.register_forward_hook(save_activation("last_mlp"))
        monosemantic_info = activations
        print("Activations:", activations)
        print(monosemantic_info)
    
    print("\n*** I'm feeling great today! Hope you're well, too. ***")


if __name__ == "__main__":
    main()
