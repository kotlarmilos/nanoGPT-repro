import torch
import torch.nn as nn
from torch.nn import functional as F

# TODO: Reduce precision to test performance
class NanoGPT(nn.Module):
  def __init__(self, vocab_size, context_size, n_embd, n_transformers):
    super().__init__()
    self.context_size = context_size
    self.n_transformers = n_transformers

    self.token_embedding = nn.Embedding(vocab_size, n_embd, dtype=torch.float32)
    self.pos_embedding = nn.Embedding(context_size, n_embd, dtype=torch.float32)
    self.blocks = nn.ModuleList([Block(n_embd // n_transformers) for _ in range(n_transformers)])
    self.dropout = nn.Dropout(0.2)
    self.lm_head = nn.Linear(n_embd, vocab_size, dtype=torch.float32)

  def forward(self, x, y=None):
    logits = self.token_embedding(x) # (B - batch,T - time/context_sie,C - channel/vocab_size)

    B, T, C = logits.shape

    pos = torch.arange(T, dtype=torch.long, device=x.device)
    pos_emb = self.pos_embedding(pos)
    pos_emb = pos_emb.unsqueeze(0)
    logits = logits + pos_emb

    chunks = logits.chunk(self.n_transformers, dim=2)
    out_chunks = [blk(chunk) for blk, chunk in zip(self.blocks, chunks)]
    logits = torch.cat(out_chunks, dim=2)

    logits = self.dropout(logits)
    logits = self.lm_head(logits)

    if y is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.contiguous().view(B*T, C)
      y = y.view(B*T)
      loss = F.cross_entropy(logits, y)

    return logits, loss

  def generate(self, x, max_new_tokens):
    for _ in range(max_new_tokens):
      x_cond = x[:, -self.context_size:]
      logits, _ = self(x_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      x_next = torch.multinomial(probs, num_samples=1)
      x = torch.cat((x, x_next), dim=1)
    return x
  
class Block(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.attention_head = AttentionHead(n_embd)
    self.layernorm1 = nn.LayerNorm(n_embd, eps=1e-5, dtype=torch.float32)
    self.feedforward = nn.Linear(n_embd, n_embd, dtype=torch.float32)
    self.layernorm2 = nn.LayerNorm(n_embd, eps=1e-5, dtype=torch.float32)

  def forward(self, x):
    x = x + self.attention_head(self.layernorm1(x))
    x = x + self.feedforward(self.layernorm2(x))
    return x
  
class AttentionHead(nn.Module):
  def __init__(self, n_embd):
      super().__init__()
      self.head_size = n_embd
      self.key = nn.Linear(n_embd, n_embd, bias=False, dtype=torch.float32)
      self.query = nn.Linear(n_embd, n_embd, bias=False, dtype=torch.float32)
      self.value = nn.Linear(n_embd, n_embd, bias=False, dtype=torch.float32)

  def forward(self, x, y=None):
      B, T, C= x.shape

      K = self.key(x)
      Q = self.query(x)
      V = self.value(x)

      dot = Q @ K.transpose(-1, -2)
      dot = dot / (self.head_size ** 0.5) # attention scores
      
      mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)).unsqueeze(0)
      dot = dot.masked_fill(~mask, float('-inf')) # fill the upper triangle with -inf
      weights = F.softmax(dot, dim=-1)
      x = weights @ V
      
      return x