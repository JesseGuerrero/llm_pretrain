
import sys
sys.path.append('.')
from prepare_data import prepare_dataset

dataset, tokenizer = prepare_dataset('./knowledge-base')
dataset.save_to_disk('./processed_dataset')
tokenizer.save_pretrained('./tokenizer')
print('Dataset preparation complete!')
