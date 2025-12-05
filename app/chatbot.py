"""
Empathetic Therapist-Style Chatbot (Multi-turn Version)
-------------------------------------------------------
- LSTM emotion classifier (local PyTorch model)
- OpenAI chat model generating therapist-style responses
- Conversation history is kept in memory so the assistant
  can respond coherently across turns.
"""

from typing import Tuple, List, Dict

from dotenv import load_dotenv
from openai import OpenAI

from .emotion_model import load_emotion_pipeline

# Load environment variables from .env (OPENAI_API_KEY)
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# -----------------------------
# 1. Label mapping
# -----------------------------
ID2LABEL = {
    0: "anger",
    1: "sadness",
    2: "joy",
    3: "neutral",
}

# -----------------------------
# 2. Load emotion model
# -----------------------------
emotion_pipeline = load_emotion_pipeline(
    checkpoint_path="models/emotion_lstm.pt",
    vocab_path="models/vocab.pkl",
    id2label=ID2LABEL,
    embed_dim=128,   # must match training
    hidden_dim=256,  # must match training
)

# -----------------------------
# 3. Keyword-based postprocess
# -----------------------------

JOY_KEYWORDS = ["happy", "excited", "glad", "grateful", "satisfied", "proud"]
SAD_KEYWORDS = ["sad", "upset", "depressed", "unhappy", "lonely", "down"]
ANGER_KEYWORDS = ["angry", "mad", "furious", "pissed", "annoyed", "irritated"]


def postprocess_emotion(user_message: str, emotion: str, confidence: float) -> Tuple[str, float]:
    """
    If the model predicts 'neutral' but the text contains strong sentiment
    keywords, adjust the emotion accordingly.
    """
    text = user_message.lower()

    if emotion == "neutral":
        if any(kw in text for kw in JOY_KEYWORDS):
            emotion = "joy"
        elif any(kw in text for kw in SAD_KEYWORDS):
            emotion = "sadness"
        elif any(kw in text for kw in ANGER_KEYWORDS):
            emotion = "anger"

    return emotion, confidence


# -----------------------------
# 4. Conversation history for OpenAI
# -----------------------------

# This will store a list of dicts:
# [{"role": "system", "content": ...},
#  {"role": "user", "content": ...},
#  {"role": "assistant", "content": ...}, ...]
CHAT_MESSAGES: List[Dict[str, str]] = []


def _ensure_system_message():
    global CHAT_MESSAGES
    if not CHAT_MESSAGES:
        system_message = (
            "You are a warm, non-judgmental therapist and coach. "
            "You speak in a gentle, reflective tone and use short paragraphs. "
            "You are having an ongoing conversation with the user. "
            "First, acknowledge and validate the user's emotions, "
            "then ask one or two open-ended questions to understand the situation better. "
            "When the user explicitly asks for help, advice, or what they should do, "
            "offer a few concrete, practical suggestions or next steps that are realistic and kind. "
            "Balance empathy with actionable guidance: summarize what you heard, "
            'then say things like “Here are a couple of options you might consider…” '
            "Avoid sounding like you're giving orders; instead, frame advice as gentle options. "
            "Do not mention that an emotion classifier is being used."
        )
        CHAT_MESSAGES.append({"role": "system", "content": system_message})


# -----------------------------
# 5. Main response generator
# -----------------------------

def generate_response(user_message: str) -> str:
    """
    Generate a therapist-style response for a single user message,
    using full conversation history so far.

    Steps:
      1. Detect emotion with local LSTM model.
      2. Append a user message (including emotion info) to CHAT_MESSAGES.
      3. Call OpenAI with the entire CHAT_MESSAGES list.
      4. Append assistant reply to history.
      5. Return reply text (optionally with emotion metadata).
    """
    global CHAT_MESSAGES

    # Step 0: make sure system message exists
    _ensure_system_message()

    # Step 1: predict emotion
    raw_emotion, confidence = emotion_pipeline.predict(user_message)
    emotion, confidence = postprocess_emotion(user_message, raw_emotion, confidence)

    # Step 2: add current user turn to history
    # We embed emotion info into the content so the model can use it,
    # but in natural language.
    user_content = (
        f"(The user's current emotional state is detected as {emotion} "
        f"with confidence {confidence:.2f}.) "
        f"The user says: {user_message}"
    )
    CHAT_MESSAGES.append({"role": "user", "content": user_content})

    # (Optional) If you worry about very long chats, you could truncate here
    # e.g., keep last N messages. For a class demo it's usually not needed.

    # Step 3: call OpenAI with full history
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=CHAT_MESSAGES,
            temperature=0.8,
            max_tokens=220,
        )
        reply_text = completion.choices[0].message.content.strip()
    except Exception as e:
        reply_text = (
            "I’m having some trouble generating a detailed response right now, "
            "but I’m still here with you. If you’d like, you can tell me a bit more "
            "about how this is affecting you.\n"
            f"(internal error: {e})"
        )
        # We don't append an error as a normal assistant turn to the history.
        return reply_text

    # Step 4: save assistant reply to history
    CHAT_MESSAGES.append({"role": "assistant", "content": reply_text})

    # Step 5: (Optional) prepend meta info for debugging.
    # If you want the web demo更自然，可以把 meta 去掉，只 return reply_text。
    meta = f"[emotion: {emotion}, confidence: {confidence:.2f}]"
    return f"{meta}\n{reply_text}"


# -----------------------------
# 6. Simple CLI loop (for testing)
# -----------------------------

def main():
    print("=== Empathetic Therapist Chatbot (LSTM + OpenAI, multi-turn) ===")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_message.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if not user_message:
            continue

        bot_reply = generate_response(user_message)
        print(f"Bot:\n{bot_reply}\n")


if __name__ == "__main__":
    main()