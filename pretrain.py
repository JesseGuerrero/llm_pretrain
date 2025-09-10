import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import os
from tqdm import tqdm


def manual_training():
    """Manual training loop that avoids Trainer framework issues."""

    print("Starting manual training...")

    # Check if dataset exists
    if not os.path.exists("./processed_dataset"):
        print("Dataset not found. Please run data preparation first.")
        return False

    # Load dataset
    dataset = load_from_disk("./processed_dataset")
    print(f"Loaded {len(dataset)} examples")

    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Spreading model across {torch.cuda.device_count()} GPUs")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically spread across GPUs
        # torch_dtype=torch.float16  # Use half precision to save memory
    )

    # Setup training
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)

    print("Starting training loop...")



    total_loss = 0
    num_steps = 0

    # Training loop
    batch_size = 8  # Process 4 examples at once across GPUs.
    print(f"Dataset length: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    for epoch in range(5):
        for i in range(0, len(dataset), batch_size):
            print(f"Processing batch starting at index {i}/{len(dataset)} for epoch {epoch}")
            batch_examples = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]

            # Prepare batch
            input_ids = [torch.tensor(ex['input_ids']) for ex in batch_examples]
            attention_masks = [torch.tensor(ex['attention_mask']) for ex in batch_examples]

            # Pad to same length
            max_len = max(len(seq) for seq in input_ids)
            input_ids = [torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)]) for seq in input_ids]
            attention_masks = [torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)]) for seq in
                               attention_masks]

            input_ids = torch.stack(input_ids).to('cuda:0')  # Send to first GPU
            attention_masks = torch.stack(attention_masks).to('cuda:0')

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=input_ids)
            loss = outputs.loss.mean()
            print(f"Raw loss: {loss.item()}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            loss.backward()
            optimizer.step()

            if (i // batch_size + 1) % 4 == 0:
                print(f"Batch {i // batch_size + 1}: Loss = {loss.item():.4f}")
            num_steps += 1
    print(f"Total steps completed: {num_steps}")

    # Save model
    output_dir = "./manual_model"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Model saved to {output_dir}")

    return True


if __name__ == "__main__":
    success = manual_training()
    if success:
        print("\nTraining successful! You can now test with the chat interface.")
    else:
        print("\nTraining failed. Check the error messages above.")