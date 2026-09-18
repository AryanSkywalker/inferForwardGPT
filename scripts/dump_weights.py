import os
import sys

import numpy as np
import torch

from transformers import GPT2Model

# check if directory 'weights' exists
# if not, create one

manifest_text = ''

pwd = os.getcwd()
weights_dir = os.path.join(pwd, 'weights')
os.makedirs(weights_dir, exist_ok=True)
print(f'Directory for weight-keeping - {weights_dir} created')

model = GPT2Model.from_pretrained('gpt2')

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
