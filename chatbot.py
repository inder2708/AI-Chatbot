import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# 1. Load the API key and make sure it exists
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("API key not found. Check that your .env file contains GEMINI_API_KEY=...")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)

# 2. Models to try, in order (a model that works moves to the front)
MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

# 3. Typing speed: seconds per character (lower = faster, 0 = instant)
TYPING_DELAY = 0.005

# 4. The personality of your chatbot (edit this freely!)
SYSTEM_PROMPT = """
You are Nova, a friendly and patient assistant.
- Speak in a warm, casual tone.
- Keep answers short (2 to 4 sentences) unless the user asks for more detail.
- If you don't know something, say so honestly instead of guessing.
- When explaining something technical, use a simple everyday example.
"""

config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)

# 5. The conversation so far (kept here so it survives a model switch)
history = []

HELP_TEXT = """Commands:
  /help   show this list
  /clear  forget the conversation and start fresh
  /save   save this chat to a file in the 'chats' folder
  quit    exit the chatbot
"""


def save_chat():
    """Write the conversation to a text file."""
    if not history:
        print("Nothing to save yet.\n")
        return

    os.makedirs("chats", exist_ok=True)
    filename = datetime.now().strftime("chats/chat_%Y-%m-%d_%H-%M-%S.txt")

    with open(filename, "w", encoding="utf-8") as f:
        for content in history:
            speaker = "You" if content.role == "user" else "Nova"
            text = "".join(part.text or "" for part in (content.parts or []))
            f.write(f"{speaker}: {text}\n\n")

    print(f"Chat saved to {filename}\n")


def ask_stream(message):
    """Send a message and print the reply as if it is being typed live."""
    global history
    for round_number in range(2):
        for model in MODELS:
            started = False
            try:
                chat = client.chats.create(model=model, config=config, history=history)

                for chunk in chat.send_message_stream(message):
                    if chunk.text:
                        if not started:
                            print("Nova: ", end="", flush=True)
                            started = True
                        for char in chunk.text:
                            print(char, end="", flush=True)
                            time.sleep(TYPING_DELAY)

                if started:
                    print("\n")
                else:
                    print("Nova: (no response, please try rephrasing)\n")

                history = chat.get_history(curated=True)

                # Remember the model that worked: move it to the front
                MODELS.remove(model)
                MODELS.insert(0, model)
                return

            except errors.ServerError:
                if started:
                    print("\n  [The reply was cut off. Please ask again.]\n")
                    return
                print(f"  ({model} is busy, trying the next model...)")

            except errors.ClientError as e:
                if started:
                    print("\n  [The reply was cut off. Please ask again.]\n")
                    return
                if e.code in (404, 429):
                    print(f"  ({model} not available right now, trying the next model...)")
                else:
                    print(f"Error: {e}\n")
                    return

            except Exception as e:
                print(f"\n  [Unexpected problem: {e}]\n  Check your internet connection and try again.\n")
                return

        time.sleep(5)

    print("Nova: Sorry, every model is busy or out of free quota right now. "
          "Please try again later.\n")


# 6. The chat loop
print("Nova is ready! Type /help for commands or 'quit' to exit.\n")

try:
    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        command = user_input.lower()

        if command in ("quit", "exit"):
            print("Goodbye!")
            break
        elif command == "/help":
            print(HELP_TEXT)
        elif command == "/clear":
            history = []
            print("Conversation cleared. Starting fresh.\n")
        elif command == "/save":
            save_chat()
        else:
            ask_stream(user_input)

except (KeyboardInterrupt, EOFError):
    print("\nGoodbye!")