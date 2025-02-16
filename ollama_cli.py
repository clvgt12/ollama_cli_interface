#!/usr/bin/env python3
"""
Ollama CLI Agent with JSON Formatted Chat History
- Maintains conversation context using a JSON list of messages.
- Each message is a dict with keys "role" and "content".
- Sends the full conversation as a prompt to the /api/generate endpoint.
- Uses prompt_toolkit for an enhanced interactive CLI.
"""

import argparse
import requests
import json
import os
import sys
from typing import List, Dict
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings

# Default values from environment variables or hardcoded defaults
DEFAULT_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "mistral:latest")


class OllamaClient:
    """
    Client for interacting with the Ollama server.
    """
    def __init__(self, host: str, model: str, debug: bool = False):
        """
        Initialize the Ollama client with server host, model, and debug flag.

        Parameters:
            host (str): The Ollama server host.
            model (str): The model name to use.
            debug (bool): Flag to enable debug mode.
        """
        self.host = host
        self.model = model
        self.debug = debug
        self.base_url = f"http://{self.host}"

    def _get_url(self, endpoint: str) -> str:
        """
        Build the full URL for a given API endpoint.

        Parameters:
            endpoint (str): The API endpoint (e.g., '/api/tags').

        Returns:
            str: The full URL.
        """
        return f"{self.base_url}{endpoint}"

    def check_connection(self) -> None:
        """
        Checks if the Ollama server is reachable.
        """
        url = self._get_url("/api/tags")
        try:
            response = requests.get(url, timeout=3)
            response.raise_for_status()
            print("✅ Successfully connected to Ollama server.")
        except requests.RequestException as e:
            print(f"❌ Error: Unable to reach Ollama server at {self.host}.")
            print("🔹 Ensure the server is running and reachable.")
            sys.exit(1)

    def verify_model(self) -> None:
        """
        Verifies if the specified model is available on the Ollama server.
        """
        url = self._get_url("/api/tags")
        try:
            response = requests.get(url, timeout=3)
            response.raise_for_status()
            models = response.json().get("models", [])
            available_models = [m["name"] for m in models]
            if self.model not in available_models:
                print(
                    f"❌ Error: Model '{self.model}' is not available on the Ollama server at {self.host}."
                )
                print(
                    f"🔹 Available models: {', '.join(available_models) if available_models else 'None'}"
                )
                sys.exit(1)
            print(f"✅ Model '{self.model}' is available on Ollama server.")
        except requests.RequestException:
            print("❌ Error: Failed to fetch models from Ollama.")
            sys.exit(1)

    def query(self, user_input: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Sends a prompt to the Ollama server by concatenating the JSON formatted chat history
        and the current user input.

        Parameters:
            user_input (str): The latest user prompt.
            chat_history (List[Dict[str, str]]): Conversation history, where each message is a dict
                                                 with keys "role" and "content".

        Returns:
            str: The assistant's complete response.
        """
        # Append the user prompt as a JSON object.
        chat_history.append({"role": "user", "content": user_input})
        
        # Build the full prompt by formatting each message as "Role: Content".
        full_prompt = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history
        )
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": True,
        }
        url = self._get_url("/api/generate")
        full_response = ""

        try:
            with requests.post(url, json=payload, stream=True) as response:
                response.raise_for_status()
                print(f"{self.model}> ", end="", flush=True)
                # Process streamed response chunks.
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            text_piece = data.get("response", "")
                            full_response += text_piece
                            print(text_piece, end="", flush=True)
                            if self.debug:
                                print(f"\n[DEBUG] Raw data: {data}")
                        except json.JSONDecodeError:
                            continue
                print("\n")
                # Append the assistant's response as a JSON object.
                chat_history.append({"role": "assistant", "content": full_response})
                return full_response
        except requests.RequestException as e:
            print(f"\nError: Unable to reach Ollama server. {e}")
            return ""


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Ollama AI Agent CLI - Interact with an AI agent using a local LLM."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Ollama server host (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode to print raw responses"
    )
    return parser.parse_args()


def setup_key_bindings() -> KeyBindings:
    """
    Sets up key bindings for the CLI interface.

    Returns:
        KeyBindings: The configured key bindings.
    """
    bindings = KeyBindings()

    @bindings.add("c-c")
    def _(event):
        print("\nExiting...")
        event.app.exit()

    return bindings


def main() -> None:
    """
    Main interactive CLI loop. Users can enter prompts and receive AI responses.
    The conversation history is maintained as a JSON list of messages and sent with each API call.
    """
    args = parse_arguments()
    client = OllamaClient(host=args.host, model=args.model, debug=args.debug)

    # Verify connection and model availability.
    client.check_connection()
    client.verify_model()

    # Warn if using a non-default model.
    if args.model != DEFAULT_MODEL:
        print(
            f"⚠️ WARNING: Using model '{args.model}' which is different from default '{DEFAULT_MODEL}'"
        )

    print(f"Connected to Ollama at {client.host}")
    print(f"Using model: {client.model}")
    if client.debug:
        print("🔍 Debug mode enabled - Printing raw responses")
    print("Welcome to the AI CLI Agent - Type your prompt and press Enter.")
    print("Type 'exit' or 'quit' to end the session.")

    bindings = setup_key_bindings()
    history = InMemoryHistory()
    # Initialize chat_history as a list of JSON objects with keys "role" and "content".
    chat_history: List[Dict[str, str]] = []

    while True:
        try:
            user_input = prompt(
                "\nYour query?> ",
                history=history,
                auto_suggest=AutoSuggestFromHistory(),
                key_bindings=bindings,
            )
            if user_input.lower() in ["exit", "quit"]:
                break
            client.query(user_input, chat_history)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
