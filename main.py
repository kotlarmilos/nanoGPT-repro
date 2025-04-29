import torch
from model import NanoGPT
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open ('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# TODO: Use tiktoken or sentencepiece
def encode(input_string):
  result = []
  for char in input_string:
    result.append(chars.index(char))
  return result

def decode(encoded_list):
  decoded_string = ""
  for index in encoded_list:
      decoded_string += chars[index]
  return decoded_string

data = torch.tensor(encode(text), dtype=torch.long)

def get_batch(batch_size, context_size):
  ix = torch.randint(len(data) - context_size, (batch_size,))
  x = torch.stack([data[i:i+context_size] for i in ix])
  y = torch.stack([data[i+1:i+context_size+1] for i in ix])
  return x, y

# Hyperparameters
batch_size = 32 # how many independent sequences to process in parallel
context_size = 256 # context size
n_embed = 512 # size of embedding vector
n_transformers = 16 # number of transformer blocks
n_iters = 15000 # number of iterations to train
learning_rate = 1e-3 # learning rate
log_interval  = 100 # how often to log training status

gpt = NanoGPT(vocab_size, context_size, n_embed, n_transformers).to(device)
optimizer = torch.optim.AdamW(gpt.parameters(), lr=learning_rate)

start = time.time()
print(decode(gpt.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))

for step in range(1, n_iters+1):
    xb, yb = get_batch(batch_size, context_size)
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = gpt(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % log_interval == 0:
        print(f"[{step}/{n_iters}] loss={loss.item():.4f} time={time.time()-start:.2f}s")
        print(decode(gpt.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))
        start = time.time()

print(decode(gpt.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))