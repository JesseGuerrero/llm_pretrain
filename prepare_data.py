import os
import glob
from datasets import Dataset
from transformers import AutoTokenizer


def collect_markdown_files(knowledge_base_path):
    """Recursively collect all markdown files from knowledge base."""
    md_files = []
    for root, dirs, files in os.walk(knowledge_base_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                md_files.append(file_path)
    return md_files


def read_markdown_content(file_paths):
    """Read content from all markdown files."""
    texts = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                print(f"\nFile: {file_path}")
                print(f"First 200 chars: {content[:200]}")  # Add this debug line
                if content:
                    texts.append(content)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return texts

def prepare_dataset(knowledge_base_path, model_name="meta-llama/Llama-3.1-8B-Instruct", max_length=512):
    """Prepare dataset for pretraining."""
    print("Collecting markdown files...")
    md_files = collect_markdown_files(knowledge_base_path)
    print(f"Found {len(md_files)} markdown files")

    print("Reading file contents...")
    texts = read_markdown_content(md_files)
    print(f"Successfully read {len(texts)} files")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize texts
    def tokenize_function(examples):
        # Tokenize and truncate to max_length
        tokens = tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
        # For causal LM, labels are the same as input_ids
        tokens['labels'] = tokens['input_ids'].copy()
        return tokens

    # Create dataset
    dataset = Dataset.from_dict({'text': texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )

    print(f"Dataset created with {len(tokenized_dataset)} examples")
    return tokenized_dataset, tokenizer


if __name__ == "__main__":
    # Example usage
    knowledge_base_path = "./knowledge_base"  # Change this to your path
    dataset, tokenizer = prepare_dataset(knowledge_base_path)

    # Save dataset for later use
    dataset.save_to_disk("./processed_dataset")
    tokenizer.save_pretrained("./tokenizer")
    print("Dataset and tokenizer saved!")