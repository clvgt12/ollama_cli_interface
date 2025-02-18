#!/usr/bin/env python3
"""
Ollama CLI Agent with YAML-Imported JSON Chat History Encapsulated in OllamaChat
- Loads an initial conversation payload from a YAML file (if specified).
- If the file cannot be loaded, creates an empty payload and optionally adds a system prompt.
- Sends the complete payload to the /api/chat endpoint and updates it with responses.
- Uses prompt_toolkit for an enhanced CLI experience.
"""

import argparse
import requests
import json
import os
import sys
import yaml  # Used for YAML file parsing
from typing import List, Dict, Optional
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings

# Default values from environment variables or hardcoded defaults.
DEFAULT_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "mistral:latest")


class OllamaClient:
    """
    Client for interacting with the Ollama server.
    Responsible only for making API calls.
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

    def send_payload(self, payload: dict) -> str:
        """
        Sends the entire conversation payload to the /api/chat endpoint.

        Parameters:
            payload (dict): The complete conversation payload.

        Returns:
            str: The assistant's complete response.
        """
        if self.debug:
            print(f"\n[DEBUG] Request payload: {payload}")
        url = self._get_url("/api/chat")
        full_response = ""
        try:
            with requests.post(url, json=payload, stream=True) as response:
                response.raise_for_status()
                print(f"{self.model}> ", end="", flush=True)
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            text_piece = data["message"]["content"]
                            full_response += text_piece
                            print(text_piece, end="", flush=True)
                            if self.debug:
                                print(f"\n[DEBUG] Response: {data}")
                        except json.JSONDecodeError:
                            continue
                print("\n")
                return full_response
        except requests.RequestException as e:
            print(f"\nError: Unable to reach Ollama server. {e}")
            return ""


class OllamaChat:
    """
    Encapsulates the chat conversation with the Ollama server.
    Manages the conversation payload and provides an interactive CLI.
    """
    def __init__(self, client: OllamaClient, prompts_file: Optional[str], system_prompt: str):
        """
        Initialize the OllamaChat session by loading an initial payload.

        Parameters:
            client (OllamaClient): The client used to communicate with the server.
            prompts_file (Optional[str]): Path to the YAML file with the initial payload.
            system_prompt (str): A system prompt to initialize conversation context.
        """
        self.client = client
        self.payload = self._load_payload(prompts_file, system_prompt)

    def _load_payload(self, prompts_file: Optional[str], system_prompt: str) -> dict:
        """
        Load the payload from a YAML file or initialize an empty payload.

        Parameters:
            prompts_file (Optional[str]): YAML file path.
            system_prompt (str): System prompt for initialization.

        Returns:
            dict: The conversation payload.
        """
        file_to_load = prompts_file if prompts_file else "prompts.yaml"
        try:
            with open(file_to_load, "r") as f:
                payload = yaml.safe_load(f)
                if payload is None:
                    payload = {}
                if not isinstance(payload, dict):
                    print(f"Warning: '{file_to_load}' does not contain a valid dictionary. Using empty payload.")
                    payload = {}
        except Exception as e:
            print(f"Warning: Could not load prompts file '{file_to_load}': {e}. Creating empty payload.")
            payload = {}

        if "model" not in payload:
            payload["model"] = self.client.model
        if "messages" not in payload or not isinstance(payload["messages"], list):
            payload["messages"] = []
        if system_prompt and not payload["messages"]:
            payload["messages"].append({"role": "system", "content": system_prompt})
        if "stream" not in payload:
            payload["stream"] = True
        return payload

    def run(self) -> None:
        """
        Runs the interactive chat loop, allowing users to send prompts and receive responses.
        """
        bindings = self._setup_key_bindings()
        history = InMemoryHistory()

        print("Welcome to the AI CLI Agent - Type your prompt and press Enter.")
        print("Type 'exit' or 'quit' to end the session.")

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

                # Update the payload with the user's message.
                self.payload["messages"].append({"role": "user", "content": user_input})
                # Send the updated payload to the API.
                assistant_response = self.client.send_payload(self.payload)
                # Update the payload with the assistant's response.
                self.payload["messages"].append({"role": "assistant", "content": assistant_response})

            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

    def _setup_key_bindings(self) -> KeyBindings:
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
        "--system",
        type=str,
        default="",
        help="A short system prompt to initialize the conversation context."
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="YAML file containing initial conversation context (default: prompts.yaml in current directory if exists)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode to print raw responses"
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the interactive CLI session.
    """
    args = parse_arguments()
    client = OllamaClient(host=args.host, model=args.model, debug=args.debug)

    # Verify connection and model availability.
    client.check_connection()
    client.verify_model()

    if args.model != DEFAULT_MODEL:
        print(
            f"⚠️ WARNING: Using model '{args.model}' which is different from default '{DEFAULT_MODEL}'"
        )

    print(f"Connected to Ollama at {client.host}")
    print(f"Using model: {client.model}")
    if client.debug:
        print("🔍 Debug mode enabled - Printing raw responses")

    # Instantiate OllamaChat with the client, prompts file, and system prompt.
    chat_session = OllamaChat(client, args.prompts, args.system)
    chat_session.run()


if __name__ == "__main__":
    main()
