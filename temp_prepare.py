import sys
sys.path.append('.')
from prepare_data import prepare_dataset

# Use HuggingFace dataset instead of local knowledge-base
dataset, tokenizer = prepare_dataset(use_huggingface=True)
dataset.save_to_disk('./processed_dataset')
tokenizer.save_pretrained('./tokenizer')
print('RuneScape dataset preparation complete!')