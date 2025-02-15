import argparse
import requests
import json
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings

# Default values
DEFAULT_HOST = "192.168.1.10:11434"
DEFAULT_MODEL = "llama3.1:8b"

# Set up argument parsing
parser = argparse.ArgumentParser(description="Ollama CLI - Interact with a local Ollama inference server.")
parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name to use (default: llama3.1:8b)")
parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Ollama server host (default: 192.168.1.10:11434)")
args = parser.parse_args()

# Assign parsed arguments
OLLAMA_HOST = args.host
MODEL_NAME = args.model
OLLAMA_API_URL = f"http://{OLLAMA_HOST}/api/generate"

# Key bindings for additional controls
bindings = KeyBindings()

@bindings.add('c-c')  # Handle Ctrl+C gracefully
def _(event):
    print("\nExiting...")
    event.app.exit()

def query_ollama(prompt_text):
    """
    Sends the prompt to the Ollama inference server and streams the response in real time.

    Args:
        prompt_text (str): The user's input to be sent to the model.

    Returns:
        None (prints output in real time)
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_text,
        "stream": True  # Enable streaming responses
    }

    try:
        with requests.post(OLLAMA_API_URL, json=payload, stream=True) as response:
            response.raise_for_status()  # Ensure the request was successful
            print(f"\n{MODEL_NAME}: ", end="", flush=True)  # Print AI prefix without newline

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        text = data.get("response", "")
                        print(text, end="", flush=True)  # Print text in real time
                    except json.JSONDecodeError:
                        continue
            print("\n")  # Newline after response is complete

    except requests.RequestException as e:
        print(f"\nError: Unable to reach Ollama server. {e}")

def main():
    """
    Main interactive CLI loop. Users can enter prompts, edit them, and receive AI responses.
    """
    history = InMemoryHistory()  # Store input history
    
    print(f"Connected to Ollama at {OLLAMA_HOST}")
    print(f"Using model: {MODEL_NAME}")
    print("Welcome to the AI CLI - Type your prompt and press Enter.")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        try:
            user_input = prompt("AI> ", history=history, auto_suggest=AutoSuggestFromHistory(), key_bindings=bindings)
            if user_input.lower() in ["exit", "quit"]:
                break
            query_ollama(user_input)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
