# starter code by matus & o1-pro
import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
    parser.set_defaults(monosemantic_enabled=False)  # disable by default

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
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        """
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


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / norm)
    
# class RotaryPositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_seq_len):
#         super(RotaryPositionalEmbedding, self).__init__()

#         self.rotation_matrix = torch.zeros(d_model, d_model, device=torch.device("cuda"))
#         for i in range(d_model):
#             for j in range(d_model):
#                 self.rotation_matrix[i, j] = math.cos(i * j * 0.01)

#         self.positional_embedding = torch.zeros(max_seq_len, d_model, device=torch.device("cuda"))
#         for i in range(max_seq_len):
#             for j in range(d_model):
#                 self.positional_embedding[i, j] = math.cos(i * j * 0.01)



    # def forward(self, x):

    #     x += self.positional_embedding[:x.size(1), :]
    #     x = torch.matmul(x, self.rotation_matrix)
    #     return x

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(RotaryPositionalEmbedding, self).__init__()

        # Initialize on CPU and then move to the appropriate device later
        self.rotation_matrix = torch.zeros(d_model, d_model)
        for i in range(d_model):
            for j in range(d_model):
                self.rotation_matrix[i, j] = math.cos(i * j * 0.01)

        self.positional_embedding = torch.zeros(max_seq_len, d_model)
        for i in range(max_seq_len):
            for j in range(d_model):
                self.positional_embedding[i, j] = math.cos(i * j * 0.01)

    def forward(self, x):
        # Move to the same device as input
        self.rotation_matrix = self.rotation_matrix.to(x.device)
        self.positional_embedding = self.positional_embedding.to(x.device)
        
        x = x + self.positional_embedding[:x.size(1), :]
        x = torch.matmul(x, self.rotation_matrix)
        return x

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
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=2, n_blocks=4, block_size = 1024, attn_dropout = 0.1, resid_dropout = 0.1, use_rope = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        if not use_rope:
            self.pos_embedding = nn.Embedding(block_size, d_model)
        else:
            self.pos_embedding = RotaryPositionalEmbedding(d_model, block_size)
        
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
        tok_emb = self.token_embedding(tokens_seq)
        
        x = self.pos_embedding(tok_emb)
      #  positions = torch.arange(0, T, dtype=torch.long, device=tokens_seq.device)
        #x = self.pos_embedding(tok_emb).unsqueeze(0)
        
      #  x = tok_emb + pos_emb
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
    return []


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
    do_monosemantic=False,
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
        
        # Get the block size limit from the model
        block_size = model.block_size if hasattr(model, 'block_size') else 1024

        for step_i in range(max_new_tokens):
            # Ensure context doesn't exceed block_size
            if len(context_tokens) >= block_size:
                # If context would be too long, truncate it (keeping the most recent tokens)
                context_tokens = context_tokens[-(block_size-1):]
                
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
    test_loader,  # New parameter
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

    train_losses = []
    test_losses = []
    steps = []

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

            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1

            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count

                model.eval()
                with torch.no_grad():
                    test_loss = 0.0
                    test_count = 0
                    for test_batch in test_loader:
                        test_batch = test_batch.to(device)
                        test_logits = model(test_batch)
                        test_batch_loss = compute_next_token_loss(test_logits, test_batch)
                        test_loss += test_batch_loss.item()
                        test_count += 1
                        if test_count >= 5:  # Limit test evaluation for speed
                            break
                    
                    avg_test_loss = test_loss / test_count if test_count > 0 else 0
                model.train()
                
                # Save for plotting
                train_losses.append(avg_part_loss)
                test_losses.append(avg_test_loss)
                steps.append(global_step)

                print(
                    f"[{model_name}] Epoch {epoch}/{epochs}, "
                    f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                    f"Train Loss: {avg_part_loss:.4f}, Test Loss: {avg_test_loss:.4f}, "
                    f"Gap: {avg_test_loss - avg_part_loss:.4f}"
                )
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
    
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(steps, train_losses, 'b-', label='Train Loss')
        plt.plot(steps, test_losses, 'r-', label='Test Loss')
        plt.title(f"{model_name} Training vs Testing Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{model_name}_overfitting.png")
        print(f"Saved overfitting analysis plot to {model_name}_overfitting.png")
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    return train_losses, test_losses, steps

def split_dataset(dataset, split_ratio=0.8):
    """Split a dataset into training and testing parts."""
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(dataset_size * split_ratio)
    
    # Custom dataset classes
    class SubsetDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, indices):
            self.dataset = original_dataset
            self.indices = indices
            
        def __len__(self):
            return len(self.indices)
            
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]
    
    # Shuffle indices
    import random
    random.shuffle(indices)
    
    # Create subsets
    train_indices = indices[:split]
    test_indices = indices[split:]
    
    train_set = SubsetDataset(dataset, train_indices)
    test_set = SubsetDataset(dataset, test_indices)
    
    return train_set, test_set


################################################################################
# 9. Main
################################################################################

def main():
    args = parse_args()
    device = torch.device(args.device_id if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparams
    embed_size = args.embed_size  
    block_size = args.block_size
    batch_size = 8
    num_epochs = 3  # Changed to 3 epochs as requested
    learning_rate = 1e-3
    log_steps = 50
    train_subset_size = 2000  # Using a small subset for faster training
    prompt = args.prompt

    # Load tokenizer
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocabulary size: {vocab_size}")

    print("Loading TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    dataset = dataset.select(range(train_subset_size))
    tinystories_seqs = [
        enc.encode(sample["text"])[:block_size]
        for sample in dataset
        if len(enc.encode(sample["text"])) > 0
    ]
    print(f"Loaded {len(tinystories_seqs)} sequences from TinyStories")

    # Create dataset with train/test split
    full_dataset = MixedSequenceDataset(tinystories_seqs, [], p_tiny=1.0)
    train_dataset, test_dataset = split_dataset(full_dataset, split_ratio=0.8)
    
    print(f"Training set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=seq_collate_fn
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=seq_collate_fn
    )

    # Define models with specified parameters
    models = {
        # KGramMLPSeqModel with k=3
        "kmlp_seq": KGramMLPSeqModel(
            vocab_size=vocab_size, 
            k=3,  # Explicitly set k=3
            embed_size=embed_size, 
            num_inner_layers=args.num_inner_mlp_layers,
            chunk_size=args.kgram_chunk_size
        ).to(device),
        
        # TransformerModel with the specified parameters
        "transformer_seq": TransformerModel(
            vocab_size=vocab_size, 
            d_model=embed_size, 
            n_heads=8, 
            n_blocks=6,
            block_size=block_size, 
            use_rope=True
        ).to(device),
    }

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n=== Training: {name} ===")
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params:,} parameters")
        
        # Train the model
        train_losses, test_losses, steps = train_one_model(
            model=model,
            loader=train_loader,
            test_loader=test_loader,
            epochs=num_epochs,
            model_name=name,
            device=device,
            lr=learning_rate,
            log_steps=log_steps,
            sample_interval=30,  # Generate samples every 30 seconds
            max_steps_per_epoch=args.max_steps_per_epoch,
            enc=enc,
            monosemantic_info=None,  # No monosemantic analysis
            prompt=prompt
        )

        print(f"\n=== Generating final samples for {name} ===")
        try:
            import matplotlib.pyplot as plt

            # Sample with different top-p values
            methods = ["Greedy", "Top-p=0.7", "Top-p=0.9", "Top-p=1.0"]
            top_p_vals = [None, 0.7, 0.9, 1.0]
            diversity = []

            for p in top_p_vals:
                samples = []
                all_tokens = []
                for _ in range(5):  # Generate 5 samples for each method
                    text, _ = generate_text(
                        model, enc, prompt, max_new_tokens=30, device=device, top_p=p
                    )
                    new_text = text[len(prompt):]
                    tokens = enc.encode(new_text)
                    samples.append(new_text)
                    all_tokens.extend(tokens)
                    
                # Calculate token diversity
                unique_ratio = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
                diversity.append(unique_ratio)
                
                print(f"[{name} | top_p={p}] Sample: {samples[0][:50]}... | Diversity: {unique_ratio:.3f}")

            # Save diversity plot with improved formatting
            plt.figure(figsize=(10, 6))
            bars = plt.bar(methods, diversity, color=["#2C3E50", "#E74C3C", "#3498DB", "#27AE60"])
            plt.title(f"Token Diversity Analysis - {name} Model", fontsize=16, fontweight='bold')
            plt.ylabel("Unique Token Ratio (Higher = More Diverse)", fontsize=12)
            plt.xlabel("Sampling Method", fontsize=12)
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
            
            # Add explanation text
            plt.figtext(0.5, 0.01, 
                      "Higher diversity indicates more varied token selection during text generation",
                      ha="center", fontsize=10, style='italic')
            
            # Improve tick labels
            plt.tick_params(axis='both', which='major', labelsize=10)
            
            # Save with high DPI for better quality
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust for the explanation text
            plt.savefig(f"{name}_token_diversity.png", dpi=300)
            print(f"Saved diversity analysis to {name}_token_diversity.png")
            plt.close()

            # Plot training and test losses with improved labels
            plt.figure(figsize=(10, 6))
            plt.plot(steps, train_losses, 'b-', linewidth=2, label='Training Loss')
            plt.plot(steps, test_losses, 'r-', linewidth=2, label='Validation Loss')
            plt.title(f"{name} Model: Training vs Validation Loss", fontsize=14)
            plt.xlabel("Training Steps", fontsize=12)
            plt.ylabel("Cross Entropy Loss", fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add min/max annotations
            min_train_idx = train_losses.index(min(train_losses))
            min_test_idx = test_losses.index(min(test_losses))
            
            plt.annotate(f'Min: {min(train_losses):.4f}', 
                         xy=(steps[min_train_idx], min(train_losses)),
                         xytext=(10, -20), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
                         
            plt.annotate(f'Min: {min(test_losses):.4f}', 
                         xy=(steps[min_test_idx], min(test_losses)),
                         xytext=(10, 20), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
            
            # Improve tick labels
            plt.tick_params(axis='both', which='major', labelsize=10)
            
            # Save with high DPI for better quality
            plt.tight_layout()
            plt.savefig(f"{name}_loss_curve.png", dpi=300)
            print(f"Saved loss curve to {name}_loss_curve.png")
            plt.close()

        except Exception as e:
            print(f"Plotting failed for {name}: {e}")

if __name__ == "__main__":
    main()
