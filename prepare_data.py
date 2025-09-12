import os
import glob
from datasets import Dataset, load_dataset
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


def read_markdown_content(file_paths, chunk_size=2048):
    """Read content from all markdown files and chunk them."""
    texts = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                print(f"\nFile: {file_path}")
                print(f"Content size: {len(content)} characters")

                if content:
                    # Split into chunks to avoid overwhelming the model
                    paragraphs = content.split('\n\n')

                    current_chunk = ""
                    for paragraph in paragraphs:
                        # If adding this paragraph would exceed chunk_size, save current chunk
                        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                            texts.append(current_chunk.strip())
                            current_chunk = paragraph
                        else:
                            current_chunk += "\n\n" + paragraph if current_chunk else paragraph

                    # Add the final chunk
                    if current_chunk.strip():
                        texts.append(current_chunk.strip())
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f"\nTotal chunks created: {len(texts)}")
    return texts


def read_single_markdown_file(file_path, chunk_size=2048):
    """Read a single large markdown file and split into chunks."""
    texts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            print(f"\nReading file: {file_path}")
            print(f"Total file size: {len(content)} characters")

            if content:
                # Split into chunks to avoid overwhelming the model
                # Split by double newlines first (paragraphs), then by size if needed
                paragraphs = content.split('\n\n')

                current_chunk = ""
                for paragraph in paragraphs:
                    # If adding this paragraph would exceed chunk_size, save current chunk
                    if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                        texts.append(current_chunk.strip())
                        current_chunk = paragraph
                    else:
                        current_chunk += "\n\n" + paragraph if current_chunk else paragraph

                # Add the final chunk
                if current_chunk.strip():
                    texts.append(current_chunk.strip())

                print(f"Split into {len(texts)} chunks")
                print(f"First chunk preview (200 chars): {texts[0][:200]}")

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return texts


def prepare_dataset(knowledge_base_path=None, model_name="EleutherAI/pythia-6.9b", max_length=512,
                    use_huggingface=False, single_file_path=None):
    """Prepare dataset for pretraining."""

    if single_file_path:
        print(f"Loading single markdown file: {single_file_path}")
        texts = read_single_markdown_file(single_file_path, chunk_size=2048)
        print(f"Loaded {len(texts)} text chunks from single file")

    elif use_huggingface:
        print("Loading RuneScape dataset from HuggingFace...")
        # Load your HuggingFace dataset
        hf_dataset = load_dataset("JesseGuerrero/2012-runescape-wiki")

        # Extract texts from the dataset - it's a CSV with 'text' column
        texts = [example["text"] for example in hf_dataset["train"]]
        print(f"Loaded {len(texts)} examples from HuggingFace dataset")

        # Debug: show first few examples
        for i, text in enumerate(texts[:3]):
            print(f"\nExample {i + 1} (first 200 chars): {text[:200]}")

    else:
        print("Collecting markdown files...")
        md_files = collect_markdown_files(knowledge_base_path)
        print(f"Found {len(md_files)} markdown files")

        print("Reading file contents...")
        texts = read_markdown_content(md_files, chunk_size=2048)
        print(f"Successfully processed {len(texts)} text chunks")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize texts
    def tokenize_function(examples):
        # Ensure all texts are strings and not None
        valid_texts = []
        for text in examples['text']:
            if isinstance(text, str) and text.strip():
                valid_texts.append(text)
            else:
                valid_texts.append("")  # Replace invalid with empty string

        # Tokenize and truncate to max_length
        tokens = tokenizer(
            valid_texts,
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
    # Example usage - use knowledge-base folder
    dataset, tokenizer = prepare_dataset(knowledge_base_path="./knowledge-base")

    # Save dataset for later use
    dataset.save_to_disk("./processed_dataset")
    tokenizer.save_pretrained("./tokenizer")
    print("Dataset and tokenizer saved!")