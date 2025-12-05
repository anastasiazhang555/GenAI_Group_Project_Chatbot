# app/web_ui.py

import gradio as gr
from .chatbot import generate_response


def respond(user_message, history_text):
    if history_text is None:
        history_text = ""

    bot_reply = generate_response(user_message)

    new_history = (
        history_text
        + f"\n🧑 You: {user_message}\n"
        + f"🤖 Bot: {bot_reply}\n"
    )

    return "", new_history


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # 🧠 Empathetic Therapist Chatbot  
    This demo combines:
    • LSTM Emotion Classifier  
    • OpenAI Therapist Persona  

    The chatbot detects user emotions and responds in a warm, reflective tone.
    """
    )

    history_area = gr.Textbox(
        label="Conversation",
        lines=20,
        interactive=False
    )
    user_input = gr.Textbox(
        placeholder="Share what's on your mind...",
        lines=2,
        label="Your message"
    )
    send_btn = gr.Button("Send")

    send_btn.click(
        respond,
        inputs=[user_input, history_area],
        outputs=[user_input, history_area]
    )

if __name__ == "__main__":
    demo.launch()