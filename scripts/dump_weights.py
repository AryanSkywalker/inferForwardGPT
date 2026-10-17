import os
import sys
import json

import numpy as np
import torch

from transformers import GPT2Model
from transformers import GPT2Tokenizer

# check if directory 'weights' exists
# if not, create one

manifest_text = ''

pwd = os.getcwd()
weights_dir = os.path.join(pwd, 'weights')
os.makedirs(weights_dir, exist_ok=True)
print(f'Directory for weight-keeping - {weights_dir} created')

model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

print('\n', "=" * 64)
for name, tensor in model.state_dict().items():
	shape = ','.join(list(map(str, tensor.shape)))

	raw_tensor = torch.flatten(tensor).detach().cpu().float().numpy()
	bin_name = name.lower().replace('.', '__')
	
	bin_dump_path = os.path.join(weights_dir, bin_name + '.kin')
	raw_tensor.tofile(bin_dump_path)

	manifest_line = f'{name} {shape} {bin_name}'
	print(f'Writing line: {manifest_line} to the manifesto')
	manifest_text += manifest_line + '\n'
print("=" * 64, '\n')

with open(os.path.join(weights_dir, "manifest.txt"), 'w', encoding='utf-8') as f:
	f.write(manifest_text)
print("Dumped weights successfully")
print("\n\n\n", "=" * 64, "\n\n\n")

vocab_file = tokenizer.vocab_file
vocab_dir = os.path.dirname(vocab_file)

print(f"Listing all files in the cached directory ---\n{os.listdir(vocab_dir)}")

merges_file = os.path.join(vocab_dir, "merges.txt")

# dumping vocab file into a file with id - hex_encoded text
with open(vocab_file, 'r', encoding='utf-8') as f:
	data = json.load(f)
	vocab_dump = ""

	for t, i in data.items():
		vocab_dump += f"{i} {t.encode('utf-8').hex()}\n"

	with open(os.path.join(weights_dir, "encoder.txt"), "w", encoding='utf-8') as df:
		df.write(vocab_dump)

# dumping merges file with hex-encoded token pairs
with open(merges_file, 'r', encoding='utf-8') as f:
	with open(os.path.join(weights_dir, "merges.txt"), "w", encoding="utf-8") as df:
		for line in f:
			parts = line.split()
			if len(parts) != 2:
				continue
			left, right = parts
			df.write(f"{left.encode('utf-8').hex()} {right.encode('utf-8').hex()}\n")
