import argparse
import requests
import json
import datetime
import os
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings

# Default values
DEFAULT_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "mistral:latest")

# Check environment variables (fallback to hardcoded defaults)
ENV_HOST = os.getenv("OLLAMA_HOST", DEFAULT_HOST)
ENV_MODEL = os.getenv("MODEL_NAME", DEFAULT_MODEL)

# Set up argument parsing
parser = argparse.ArgumentParser(description="Ollama AI Agent CLI - Interact with an AI agent using a local LLM.")
parser.add_argument("--model", type=str, default=ENV_MODEL, help=f"Model name to use (default: {ENV_MODEL})")
parser.add_argument("--host", type=str, default=ENV_HOST, help=f"Ollama server host (default: {ENV_HOST})")
parser.add_argument("--debug", action="store_true", help="Enable debug mode to print raw responses")
args = parser.parse_args()

# Assign parsed arguments
OLLAMA_HOST = args.host
MODEL_NAME = args.model
DEBUG_MODE = args.debug

# Connection Check: Ensure Ollama server is reachable
def check_ollama_connection():
    """
    Checks if the Ollama server is reachable before starting.
    """
    OLLAMA_API_URL = f"http://{OLLAMA_HOST}/api/tags"
    try:
        response = requests.get(OLLAMA_API_URL, timeout=3)
        response.raise_for_status()
        print("✅ Successfully connected to Ollama server.")
    except requests.RequestException as e:
        print(f"❌ Error: Unable to reach Ollama server at {OLLAMA_HOST}.")
        print("🔹 Ensure the server is running and reachable.")
        exit(1)  # Exit with an error status

# -------------------- Model Check -------------------- #
def verify_ollama_model(model_name: str) -> bool:
    """
    Verifies if the specified model is available on Ollama.
    """
    OLLAMA_API_URL = f"http://{OLLAMA_HOST}/api/tags"
    try:
        response = requests.get(OLLAMA_API_URL, timeout=3)
        response.raise_for_status()
        models = response.json().get("models", [])

        # Extract model names from response
        available_models = [m["name"] for m in models]

        if model_name in available_models:
            print(f"✅ Model '{model_name}' is available on Ollama server.")
            return True
        else:
            print(f"❌ Error: Model '{model_name}' is NOT available on Ollama server at {OLLAMA_HOST}.")
            print(f"🔹 Available models: {', '.join(available_models) if available_models else 'None'}")
            exit(1)  # Exit with an error status
    except requests.RequestException as e:
        print(f"❌ Error: Failed to fetch models from Ollama.")
        exit(1)  # Exit with an error status

# Perform the connection check before proceeding
check_ollama_connection()
verify_ollama_model(MODEL_NAME)

# Print a warning if the user-specified model is different from the default
if MODEL_NAME != DEFAULT_MODEL:
    print(f"⚠️ WARNING: You are using a '{MODEL_NAME}' different from the default '{DEFAULT_MODEL}'. Ensure compatibility!")

# Define key bindings for CLI
bindings = KeyBindings()

@bindings.add('c-c')  # Handle Ctrl+C gracefully
def _(event):
    print("\nExiting...")
    event.app.exit()

# -------------------- Ollama API Interaction -------------------- #
def query_ollama(prompt_text):
    """
    Sends a prompt to the Ollama inference server, correctly formatting it for raw mode function calling.
    """
    OLLAMA_API_URL = f"http://{OLLAMA_HOST}/api/generate"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_text,
        "stream": True,
    }

    try:
        with requests.post(OLLAMA_API_URL, json=payload, stream=True) as response:
            response.raise_for_status()
            print(f"\n{MODEL_NAME}: ", end="", flush=True)

            full_response = ""  # Store accumulated response

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        print(data.get("response", ""), end="", flush=True)
                    except json.JSONDecodeError:
                        continue
            print("\n")

    except requests.RequestException as e:
        print(f"\nError: Unable to reach Ollama server. {e}")

# -------------------- CLI Main Loop -------------------- #
def main():
    """
    Main interactive CLI loop. Users can enter prompts, edit them, and receive AI responses.
    """
    history = InMemoryHistory()  # Store input history

    print(f"Connected to Ollama at {OLLAMA_HOST}")
    print(f"Using model: {MODEL_NAME}")
    if DEBUG_MODE:
        print("🔍 Debug mode enabled - Printing raw responses")
    print("Welcome to the AI CLI Agent - Type your prompt and press Enter.")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        try:
            user_input = prompt("\nAI> ", history=history, auto_suggest=AutoSuggestFromHistory(), key_bindings=bindings)
            if user_input.lower() in ["exit", "quit"]:
                break
            query_ollama(user_input)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
