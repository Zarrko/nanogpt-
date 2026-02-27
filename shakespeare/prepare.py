import os 
import requests
import tiktoken
import numpy as np

# Download the shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
if not os.path.exists(input_file_path):
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open(input_file_path, "w") as f:
        f.write(requests.get(data_url).text)

# Read the dataset
with open(input_file_path, "r") as f:
    data = f.read()
print(f"Dataset length: {len(data)} characters")

n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# Tokenization with tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
train_ids = tokenizer.encode(train_data)
val_ids = tokenizer.encode(val_data)
print(f"Train has {len(train_ids)} tokens")
print(f"Val has {len(val_ids)} tokens")

# Save the tokenized data as numpy arrays
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))