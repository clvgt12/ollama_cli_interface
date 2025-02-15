#!/usr/bin/env python3
"""
Ollama CLI Agent with Chat History Concatenation
- Maintains conversation context by concatenating chat history.
- Sends the full conversation as a prompt to the /api/generate endpoint.
- Uses prompt_toolkit for an enhanced interactive CLI.
"""

import argparse
import requests
import json
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
parser = argparse.ArgumentParser(
    description="Ollama AI Agent CLI - Interact with an AI agent using a local LLM."
)
parser.add_argument(
    "--model",
    type=str,
    default=ENV_MODEL,
    help=f"Model name to use (default: {ENV_MODEL})",
)
parser.add_argument(
    "--host",
    type=str,
    default=ENV_HOST,
    help=f"Ollama server host (default: {ENV_HOST})",
)
parser.add_argument(
    "--debug", action="store_true", help="Enable debug mode to print raw responses"
)
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
        exit(1)

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
        available_models = [m["name"] for m in models]

        if model_name in available_models:
            print(f"✅ Model '{model_name}' is available on Ollama server.")
            return True
        else:
            print(
                f"❌ Error: Model '{model_name}' is NOT available on Ollama server at {OLLAMA_HOST}."
            )
            print(
                f"🔹 Available models: {', '.join(available_models) if available_models else 'None'}"
            )
            exit(1)
    except requests.RequestException as e:
        print("❌ Error: Failed to fetch models from Ollama.")
        exit(1)

# Perform the connection check before proceeding
check_ollama_connection()
verify_ollama_model(MODEL_NAME)

# Print a warning if the user-specified model is different from the default
if MODEL_NAME != DEFAULT_MODEL:
    print(
        f"⚠️ WARNING: You are using the model '{MODEL_NAME}' which is different from the default model '{DEFAULT_MODEL}'"
    )

# Define key bindings for CLI
bindings = KeyBindings()

@bindings.add("c-c")  # Handle Ctrl+C gracefully
def _(event):
    print("\nExiting...")
    event.app.exit()

# -------------------- Ollama API Interaction with Concatenated Chat History -------------------- #
def query_ollama(user_input, chat_history):
    """
    Sends a prompt to the Ollama inference server by concatenating chat history and the current user input.

    Parameters:
      user_input (str): The latest user prompt.
      chat_history (list): List containing the conversation history.
                         Each entry is a string prefixed with 'User:' or 'Assistant:'.

    Returns:
      str: The complete assistant response.
    """
    # Append the new user prompt to the chat history with a role indicator.
    chat_history.append(f"User: {user_input}")

    # Concatenate the entire chat history into a single prompt.
    full_prompt = "\n".join(chat_history)

    # Prepare the payload for the /api/generate endpoint.
    OLLAMA_API_URL = f"http://{OLLAMA_HOST}/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": True,
    }

    try:
        with requests.post(OLLAMA_API_URL, json=payload, stream=True) as response:
            response.raise_for_status()
            print(f"{MODEL_NAME}> ", end="", flush=True)
            full_response = ""

            # Process streamed response chunks.
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        text_piece = data.get("response", "")
                        full_response += text_piece
                        print(text_piece, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
            print("\n")

            # Append the assistant's response to the chat history.
            chat_history.append(f"Assistant: {full_response}")

            return full_response
    except requests.RequestException as e:
        print(f"\nError: Unable to reach Ollama server. {e}")
        return ""

# -------------------- CLI Main Loop -------------------- #
def main():
    """
    Main interactive CLI loop. Users can enter prompts, edit them, and receive AI responses.
    The conversation history is concatenated and sent with each API call.
    """
    history = InMemoryHistory()  # Store input history
    chat_history = []            # Maintains the full conversation context

    print(f"Connected to Ollama at {OLLAMA_HOST}")
    print(f"Using model: {MODEL_NAME}")
    if DEBUG_MODE:
        print("🔍 Debug mode enabled - Printing raw responses")
    print("Welcome to the AI CLI Agent - Type your prompt and press Enter.")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        try:
            user_input = prompt(
                "\nYour query?> ", history=history, auto_suggest=AutoSuggestFromHistory(), key_bindings=bindings
            )
            if user_input.lower() in ["exit", "quit"]:
                break
            query_ollama(user_input, chat_history)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
