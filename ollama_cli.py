#!/usr/bin/env python3
"""
Ollama CLI Agent with YAML-Imported JSON Chat History and Agentic Tool Execution
- Loads an initial conversation payload from a YAML file (if specified or found).
- Merges configuration values for host, model, and system prompt from command line,
  environment variables, YAML file, and defaults.
- Sends the complete payload to the /api/chat endpoint and updates it with responses.
- Uses prompt_toolkit for an enhanced CLI experience.
- Dynamically executes tool functions when the LLM indicates to do so.
"""

import argparse
import requests
import json
import os
import re
import sys
import yaml  # Used for YAML file parsing
import logging
from typing import Optional
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings

# Default values (lowest precedence)
DEFAULT_HOST = "localhost:11434"
DEFAULT_MODEL = "mistral:latest"
DEFAULT_SYSTEM = "you are a helpful assistant."

# --------------------------
# Global Tool Registry Setup
# --------------------------

# Global registry for tool functions.
TOOL_REGISTRY = {}

def register_tool(name):
    """
    Slogan: Registers a tool function in the global tool registry.
    Parameters:
        name (str): The name of the tool function.
    Returns:
        function: Decorator that registers the function.
    """
    def decorator(func):
        TOOL_REGISTRY[name] = func
        return func
    return decorator

def execute_tool_function(tool_call: dict):
    """
    Executes a tool function based on the provided tool call.
    
    This function extracts the tool function's name and its arguments from the
    provided dictionary under the "function" key. It then retrieves the corresponding
    function from the TOOL_REGISTRY and executes it with the given arguments.
    
    Parameters:
        tool_call (dict): A dictionary containing a "function" key. The associated value 
                          should be a dictionary with:
                              - "name" (str): The name of the tool function to execute.
                              - "arguments" (dict): A dictionary of arguments for the tool function.
    
    Returns:
        dict: A JSON-like dictionary with the following keys:
                - "tool": (str) The name of the executed tool function.
                - "content": (Any) The result of executing the tool function. In case of an error,
                             this will contain an error message.
    """
    # Retrieve the 'function' sub-dictionary
    function_data = tool_call.get("function", {})
    
    # Extract the function's name and arguments
    function_name = function_data.get("name", "")
    function_args = function_data.get("arguments", {})

    # Retrieve the tool function from the registry
    tool_function = TOOL_REGISTRY.get(function_name)
    if not tool_function:
        error_msg = f"Tool '{function_name}' is not registered."
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Execute the tool function with the provided arguments.
        # NOTE: We use json.dumps() to serialize the result to a JSON string if needed.
        result = {"role": "tool", "content": tool_function(**function_args)}
        return result
    except Exception as e:
        logging.error("Error executing tool '%s': %s", function_name, e)
        result = {"role": "tool", "content": f"Error executing tool: {e}"}
        return result

# --------------------------
# Sample Tool Implementation
# --------------------------

@register_tool("get_current_weather")
def get_current_weather(location: str, format: str) -> str:
    """
    Slogan: Retrieves current weather information for a location.
    Parameters:
        location (str): The location to get the weather for.
        format (str): Temperature format, either 'celsius' or 'fahrenheit'.
    Returns:
        str: A string representing the current weather.
    """
    if format == "celsius":
        return f"Current weather in {location}: 22°C with clear skies."
    else:
        return f"Current weather in {location}: 72°F with clear skies."

# --------------------------
# Configuration Merging Logic
# --------------------------

def merge_config(args: argparse.Namespace) -> dict:
    """
    Slogan: Merges configuration from command line, environment, YAML, and defaults.
    Parameters:
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        dict: Merged configuration with keys: host, model, system, prompts_file.
    """
    # Attempt to load YAML configuration if a file is specified or if "prompts.yaml" exists.
    yaml_config = {}
    filename = None
    if args.prompts:
        filename = args.prompts
    elif os.path.exists("prompts.yaml"):
        filename = "prompts.yaml"
    
    if filename:
        try:
            with open(filename, "r") as f:
                yaml_config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load prompts file '{filename}'")
    
    # Extract values from YAML file (if available)
    yaml_host = yaml_config.get("host")
    yaml_model = yaml_config.get("model") or yaml_config.get("model_name")
    yaml_system = yaml_config.get("system") or yaml_config.get("system_prompt")
    
    # Environment variables
    env_host = os.getenv("OLLAMA_HOST")
    env_model = os.getenv("MODEL_NAME")
    
    # Merge using precedence: command line > environment > YAML > default.
    final_host = (args.host if args.host is not None 
                  else (env_host if env_host is not None 
                        else (yaml_host if yaml_host is not None 
                              else DEFAULT_HOST)))
    
    final_model = (args.model if args.model is not None 
                   else (env_model if env_model is not None 
                         else (yaml_model if yaml_model is not None 
                               else DEFAULT_MODEL)))
    
    final_system = (args.system if args.system is not None 
                    else (yaml_system if yaml_system is not None 
                          else DEFAULT_SYSTEM))
    
    return {
        "host": final_host,
        "model": final_model,
        "system": final_system,
        "prompts_file": filename  # Either user-specified or discovered "prompts.yaml"
    }

# --------------------------
# Ollama Client and Chat Classes
# --------------------------

class OllamaClient:
    """
    Client for interacting with the Ollama server.
    Responsible only for making API calls.
    """
    def __init__(self, host: str, model: str, debug: bool = False):
        """
        Slogan: Initializes the Ollama client.
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

    def _process_tool_calls(self, data: dict) -> str:
        """
        Slogan: Processes tool_calls object and calls functions.
        Parameters:
            data (dict): The complete response.
        Returns:
            tool_msgs(list of dict): A list of tool messages results from successive tool function calls
        """
        # Create an empty list to accumulate tool_msgs when processing the array tool_calls
        tool_msgs = []
        # Retrieve the 'message' sub-dictionary; use an empty dict if not present
        message = data.get("message", {})
        # Retrieve the 'tool_calls' array if it exists; otherwise, set to None
        tool_calls = message.get("tool_calls",{})
        if(tool_calls is not None):
            for tool_call in tool_calls:
                # Process assistant response for potential tool calls.
                tool_msgs.append(execute_tool_function(tool_call)) 
                if self.debug:
                    print(f"[DEBUG OllamaClient]: tool_msgs:{tool_msgs}")
        return tool_msgs

    def send_payload(self, payload: dict) -> str:
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
                            tool_response = self._process_tool_calls(data)
                            if tool_response is not None:
                                for t in tool_response:
                                    messages.append(t)
                            if self.debug:
                                print(f"\n[DEBUG OllamaClient] Response: {data}, tool_response: {tool_response}")
                        except json.JSONDecodeError:
                            continue
                print("\n")
                if full_text is not None:
                    messages.append({"role": "assistant", "content": full_text})
                return messages
        except requests.RequestException as e:
            print(f"\nError: Unable to reach Ollama server. {e}")
            return []

class OllamaChat:
    """
    Encapsulates the chat conversation with the Ollama server.
    Manages the conversation payload and provides an interactive CLI.
    """
    def __init__(self, client: OllamaClient, prompts_file: Optional[str], system_prompt: str, debug: bool = False):
        """
        Slogan: Initializes the OllamaChat session.
        Parameters:
            client (OllamaClient): The client for server communication.
            prompts_file (Optional[str]): Path to the YAML file with the initial payload.
            system_prompt (str): A system prompt to initialize the conversation context.
            debug (bool): Flag to enable debug mode.
        Returns: None.
        """
        self.client = client
        self.payload = self._load_payload(prompts_file, system_prompt)
        self.debug = debug

    def _load_payload(self, prompts_file: Optional[str], system_prompt: str) -> dict:
        """
        Slogan: Loads the payload from a YAML file or initializes an empty payload.
        Parameters:
            prompts_file (Optional[str]): YAML file path.
            system_prompt (str): System prompt for initialization.
        Returns:
            dict: The conversation payload.
        """
        file_to_load = prompts_file if prompts_file else "prompts.yaml"
        try:
            with open(file_to_load, "r") as f:
                payload = yaml.safe_load(f) or {}
                if not isinstance(payload, dict):
                    print(f"Warning: '{file_to_load}' does not contain a valid dictionary. Using empty payload.")
                    payload = {}
        except Exception as e:
            print(f"Warning: Could not load prompts file '{file_to_load}': {e}. Creating empty payload.")
            payload = {}

        # Merge configuration into payload if not already set.
        payload["model"] = (self.client.model if self.client.model is not None 
                       else (payload.get("model")))
        payload.setdefault("messages", [])
        payload.setdefault("stream", True)
        if system_prompt and not payload["messages"]:
            payload["messages"].append({"role": "system", "content": system_prompt})
        
        # Optionally validate tools defined in the YAML.
        if "tools" in payload:
            for tool in payload["tools"]:
                tool_name = tool.get("function", {}).get("name")
                if tool_name and tool_name not in TOOL_REGISTRY:
                    print(f"Warning: Tool '{tool_name}' defined in YAML is not registered in the agent.")
        
        return payload
            
    def run(self) -> None:
        """
        Slogan: Runs the interactive chat loop.
        Parameters: None.
        Returns: None.
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

                # Append user's message.
                self.payload["messages"].append({"role": "user", "content": user_input})
                # Send payload to the API.
                messages = self.client.send_payload(self.payload)
                # Append assistant messages to payload messages to maintain chat memory.
                for m in messages:
                    self.payload["messages"].append(m)
                if self.debug:
                    print(f"[DEBUG OllamaChat]: messages: {messages}")
                    print(f"[DEBUG OllamaChat]: payload: {self.payload}")

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
        help="Model name to use (overrides environment and YAML)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Ollama server host (overrides environment and YAML)"
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="A short system prompt to initialize the conversation context (overrides YAML)"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="YAML file containing initial conversation context (default: prompts.yaml if exists)"
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
    
    client = OllamaClient(host=config["host"], model=config["model"], debug=args.debug)

    # Verify connection and model availability.
    client.check_connection()
    client.verify_model()

    print(f"Connected to Ollama at {client.host}")
    print(f"Using model: {client.model}")
    if args.debug:
        print("🔍 Debug mode enabled - Printing raw responses")

    # Instantiate OllamaChat with the merged system prompt and prompts file.
    chat_session = OllamaChat(client, config["prompts_file"], config["system"], debug=args.debug)
    chat_session.run()

if __name__ == "__main__":
    main()
