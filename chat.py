import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr


def load_model(model_path="./manual_model"):
    """Load the trained model."""
    print("Loading model...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        print(f"Model loaded from {model_path} on {device}")
        return model, tokenizer, device

    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")

        # Try simple_model path
        try:
            print("Trying ./simple_model...")
            tokenizer = AutoTokenizer.from_pretrained("./simple_model")
            model = AutoModelForCausalLM.from_pretrained("./simple_model")

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            print(f"Model loaded from ./simple_model on {device}")
            return model, tokenizer, device

        except Exception as e2:
            print(f"Failed to load from ./simple_model: {e2}")
            print("Falling back to original model...")

            # Fallback to original model
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            print(f"Using original model on {device}")
            return model, tokenizer, device


def generate_response(model, tokenizer, device, prompt, max_length=100):
    """Generate a response from the model."""
    try:
        # Encode the prompt
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                top_p=0.9
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the new part
        new_text = response[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()

        return new_text if new_text else "I'm not sure how to respond to that."

    except Exception as e:
        return f"Error generating response: {str(e)}"


def create_interface():
    """Create Gradio interface."""

    # Load model once
    model, tokenizer, device = load_model()

    def chat_fn(message, history):
        """Handle chat interaction."""
        if not message.strip():
            return history, ""

        # Generate response
        response = generate_response(model, tokenizer, device, message)

        # Add to history
        history = history or []
        history.append((message, response))

        return history, ""

    # Create interface
    with gr.Blocks(title="Simple Knowledge Chat") as demo:
        gr.Markdown("# Simple Knowledge Base Chat")
        gr.Markdown("Test your fine-tuned model here!")

        chatbot = gr.Chatbot(label="Conversation", height=400)
        msg = gr.Textbox(
            placeholder="Type your message here...",
            label="Your message",
            lines=2
        )

        submit = gr.Button("Send")
        clear = gr.Button("Clear")

        # Event handlers
        submit.click(chat_fn, [msg, chatbot], [chatbot, msg])
        msg.submit(chat_fn, [msg, chatbot], [chatbot, msg])
        clear.click(lambda: ([], ""), outputs=[chatbot, msg])

    return demo

from pyngrok import ngrok
if __name__ == "__main__":
    ngrok.set_auth_token("2aveDiR2bbXKfz8kkHaQIzh1AvO_7kWS54CPDiG3Day3nUJR4")
    print("Starting simple chat interface...")
    demo = create_interface()
    public_url = ngrok.connect(7860)
    print(f"Public URL: {public_url}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )