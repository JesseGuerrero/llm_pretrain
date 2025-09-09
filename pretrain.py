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
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Setup training
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)

    print("Starting training loop...")

    total_loss = 0
    num_steps = 0

    # Training loop
    for epoch in range(1):  # 1 epoch
        print(f"Epoch {epoch + 1}")

        for i, example in enumerate(tqdm(dataset, desc="Training")):
            # Prepare inputs
            input_ids = torch.tensor([example['input_ids']], dtype=torch.long).to(device)
            attention_mask = torch.tensor([example['attention_mask']], dtype=torch.long).to(device)
            labels = torch.tensor([example['input_ids']], dtype=torch.long).to(device)  # For causal LM

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track progress
            total_loss += loss.item()
            num_steps += 1

            # Log progress
            if (i + 1) % 10 == 0:
                avg_loss = total_loss / num_steps
                print(f"Step {i + 1}/{len(dataset)}: Loss = {loss.item():.4f}, Avg Loss = {avg_loss:.4f}")

    # Save model
    output_dir = "./manual_model"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✓ Training completed! Final average loss: {total_loss / num_steps:.4f}")
    print(f"✓ Model saved to {output_dir}")

    return True


if __name__ == "__main__":
    success = manual_training()
    if success:
        print("\nTraining successful! You can now test with the chat interface.")
    else:
        print("\nTraining failed. Check the error messages above.")