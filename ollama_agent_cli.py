#!/usr/bin/env python3
"""
Ollama CLI Agent with Agentic Tool Execution using smolagents library.
- Merges configuration values for inference host and model from command line.
- Leverages smolagents library to manage the agentic ReAct cycle and maintain chat context.
- Uses prompt_toolkit for an enhanced CLI experience.
"""

import argparse
import requests
import json
import os
import re
import sys
import logging
from typing import Optional, List, Dict, Any
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from smolagents import CodeAgent, LiteLLMModel, tool

# Default values (lowest precedence)
DEFAULT_HOST = "localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:14b"
DEFAULT_URL = f"http://{DEFAULT_HOST}"

# --------------------------
# Tool Setup
# --------------------------

@tool
def get_current_weather(location: str, temp_format: str) -> str:
    """
    Retrieves current weather forecast for a location using live API calls.
    
    Args:
        location: The location in the format "city,state,country" (state is optional).
        temp_format: Temperature format, either 'celsius' or 'fahrenheit'.
    """
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        logging.error("API key for OpenWeatherMap is not set in the environment variable 'OPENWEATHER_API_KEY'.")
        raise ValueError("Missing API key for OpenWeatherMap.")

    # Map the temperature format to API units.
    units = "metric" if temp_format.lower() == "celsius" else "imperial"
    
    # Get the weather forecast JSON object.
    if location is not None and location.strip() != "":
        params = {"q": location.strip(), "units": units, "appid": api_key}
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            weather_json = response.json()
            return json.dumps(weather_json)
        except requests.RequestException as e:
            return(f"I could not obtain the current weather data for {location}. Can you enter it again?")

    else:
        return (f"""I didn't understand the location you provided. Can you enter it again?""")

# --------------------------
# Configuration Merging Logic
# --------------------------

def merge_config(args: argparse.Namespace) -> dict:
    """
    Slogan: Merges configuration from command line, environment, and defaults.
    Parameters:
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        dict: Merged configuration with keys: host, model.
    """
    
    # Environment variables
    env_host = os.getenv("OLLAMA_HOST")
    env_model = os.getenv("MODEL_NAME")
    
    # Merge using precedence: command line > environment > default.
    final_host = (args.host if args.host is not None 
                  else (env_host if env_host is not None 
                            else DEFAULT_HOST))
    
    final_url = (f"http://{final_host}" if final_host is not None
                    else DEFAULT_URL)
    
    final_model = (args.model if args.model is not None 
                   else (env_model if env_model is not None 
                            else DEFAULT_MODEL))
    
    return {
        "host": final_host,
        "model": final_model,
        "url": final_url,
    }

# --------------------------
# Ollama Client and Chat Classes
# --------------------------

class OllamaClient:
    """
    Client for interacting with the Ollama server.
    Responsible only for making API calls.
    """
    def __init__(self, host: str, model: str, url: str, debug: bool = False):
        """
        Slogan: Initializes the Ollama client.
        Parameters:
            host (str): The Ollama server host.
            model (str): The model name to use.
            url (str): The Ollama server URL
            debug (bool): Flag to enable debug mode.
        """
        self.host = host
        self.model = model
        self.debug = debug
        self.base_url = url

    def _get_url(self, endpoint: str) -> str:
        """
        Slogan: Builds the full URL for an API endpoint.
        Parameters:
            endpoint (str): The API endpoint (e.g., '/api/tags').
        Returns:
            str: The full URL.
        """
        return f"{self.base_url}{endpoint}"

    def check_connection(self) -> None:
        """
        Slogan: Checks if the Ollama server is reachable.
        Parameters: None.
        Returns: None.
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
        Slogan: Verifies the specified model is available on the Ollama server.
        Parameters: None.
        Returns: None.
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

    def send_payload(self, payload: dict) -> List[Dict[str, Any]]:
        """
        Slogan: Sends the conversation payload to the /api/chat endpoint.
        Parameters:
            payload (dict): The complete conversation payload.
        Returns:
            messages (list of dict): The assistant's messages to be preserved in payload messages for chat memory.
        """
        if self.debug:
            print(f"\n[DEBUG OllamaClient] Request payload: {payload}")
        url = self._get_url("/api/chat")
        full_text = ""
        messages = []
        try:
            with requests.post(url, json=payload, stream=True) as response:
                response.raise_for_status()
                print(f"{self.model}> ", end="", flush=True)
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            text_piece = data["message"]["content"]
                            full_text += text_piece
                            print(text_piece, end="", flush=True)
                            if self.debug:
                                print(f"\n[DEBUG OllamaClient] Response: {data}")
                        except json.JSONDecodeError:
                            continue
                print("\n")
                if full_text is not None:
                    messages.append({"role": "assistant", "content": full_text})
                return messages
        except requests.RequestException as e:
            logging.error(f"\n[ERROR send_payload]:: Unable to reach Ollama server. {e}")
            return []

class SmolAgentsChat:
    """
    Encapsulates the chat conversation with the Ollama server using the smolagents library.
    Manages the conversation payload and provides an interactive CLI.
    """
    def __init__(self, client, debug: bool = False):
        """
        Initializes the Chat session with a provided client.
        
        Parameters:
            client: The OllamaClient instance.
            debug (bool): Flag to enable debug mode.
        """
        self.client = client
        self.debug = debug
            
    def run(self) -> None:
        """
        Slogan: Runs the interactive chat loop.
        Parameters: None.
        Returns: None.
        """
        bindings = self._setup_key_bindings()
        history = InMemoryHistory()

        engine = LiteLLMModel(
            model_id=f"ollama/{self.client.model}",
            api_base=self.client.base_url,
        )

        agent = CodeAgent(
            tools=[get_current_weather], 
            model=engine, 
            additional_authorized_imports=['json','requests','os'],
            planning_interval=1 # This is where you activate planning!
        )

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
                if(user_input != ''):
                    result = agent.run(user_input)

            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

    def _setup_key_bindings(self) -> KeyBindings:
        """
        Slogan: Sets up CLI key bindings.
        Parameters: None.
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
    Slogan: Parses command-line arguments.
    Parameters: None.
    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Ollama AI Agent CLI - Interact with an AI agent using a local LLM."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (overrides environment)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Ollama server host (overrides environment)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode to print raw responses"
    )
    return parser.parse_args()

def main() -> None:
    """
    Slogan: Main entry point for the interactive CLI session.
    Parameters: None.
    Returns: None.
    """
    args = parse_arguments()
    config = merge_config(args)

    client = OllamaClient(host=config["host"], model=config["model"], url=config["url"], debug=args.debug)

    # Verify connection and model availability.
    client.check_connection()
    client.verify_model()

    print(f"Connected to Ollama at {client.host}")
    print(f"Using model: {client.model}")
    if args.debug:
        print("🔍 Debug mode enabled - Printing raw responses")

    # Instantiate SmolAgentsChat.
    chat_session = SmolAgentsChat(client, debug=args.debug)
    chat_session.run()

if __name__ == "__main__":
    main()
