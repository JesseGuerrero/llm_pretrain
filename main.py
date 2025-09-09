#!/usr/bin/env python3
"""
Final working pipeline - Manual training to avoid framework issues.
"""

import os
import sys

from pyngrok import ngrok
def main():
    print("Final LLM Knowledge Base Training")
    print("=" * 40)

    # Get knowledge base path
    if len(sys.argv) > 1:
        kb_path = sys.argv[1]
    else:
        kb_path = input("Enter knowledge base folder path: ").strip()

    if not os.path.exists(kb_path):
        print(f"Error: Path '{kb_path}' not found!")
        return

    print(f"Using knowledge base: {kb_path}")

    # Step 1: Prepare data (reuse existing if available)
    if not os.path.exists('./processed_dataset'):
        print("\n1. Preparing data...")
        try:
            from prepare_data import prepare_dataset
            dataset, tokenizer = prepare_dataset(kb_path)
            dataset.save_to_disk('./processed_dataset')
            tokenizer.save_pretrained('./tokenizer')
            print("✓ Data ready!")
        except Exception as e:
            print(f"✗ Data prep failed: {e}")
            return
    else:
        print("\n1. Using existing processed dataset...")

    # Step 2: Manual training (avoids Trainer framework issues)
    print("\n2. Running manual training...")
    try:
        import pretrain
        success = pretrain.manual_training()
        if success:
            print("✓ Training completed!")
        else:
            print("✗ Training had issues, but may have saved a partial model")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        # Continue anyway - we might have saved a partial model

    # Step 3: Test with chat
    print("\n3. Starting chat interface...")
    print("Chat interface will open at: http://localhost:7860")
    print("Press Ctrl+C to stop")

    try:
        import chat
        ngrok.set_auth_token("2aveDiR2bbXKfz8kkHaQIzh1AvO_7kWS54CPDiG3Day3nUJR4")
        demo = chat.create_interface()
        public_url = ngrok.connect(7860)
        print(f"Public URL: {public_url}")
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    except KeyboardInterrupt:
        print("\nChat stopped by user")
    except Exception as e:
        print(f"Chat failed: {e}")
        print("You can try running: python simple_chat.py")


if __name__ == "__main__":
    main()