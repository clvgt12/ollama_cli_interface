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
    Slogan: Executes a tool function based on the provided tool call.
    Parameters:
        tool_call (dict): Contains keys "tool" (str) and "args" (dict).
    Returns:
        Any: The result of executing the tool function.
    """
    tool_name = tool_call.get("tool")
    args = tool_call.get("args", {})
    tool_func = TOOL_REGISTRY.get(tool_name)
    if not tool_func:
        error_msg = f"Tool '{tool_name}' is not registered."
        logging.error(error_msg)
        raise ValueError(error_msg)
    try:
        return tool_func(**args)
    except Exception as e:
        logging.error("Error executing tool '%s': %s", tool_name, e)
        return f"Error executing tool '{tool_name}': {e}"

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

    def send_payload(self, payload: dict) -> str:
        """
        Slogan: Sends the conversation payload to the /api/chat endpoint.
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
        Slogan: Initializes the OllamaChat session.
        Parameters:
            client (OllamaClient): The client for server communication.
            prompts_file (Optional[str]): Path to the YAML file with the initial payload.
            system_prompt (str): A system prompt to initialize the conversation context.
        Returns: None.
        """
        self.client = client
        self.payload = self._load_payload(prompts_file, system_prompt)

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

    def extract_json_from_text(self, text: str) -> Optional[str]:
        """
        Slogan: Extracts the first JSON object substring from text.
        Parameters:
            text (str): The input text that may contain a JSON object.
        Returns:
            Optional[str]: The extracted JSON substring if found, else None.
        """
        match = re.search(r'({.*})', text, re.DOTALL)
        if match:
            return match.group(1)
        return None

    def fix_invalid_json(self, json_str: str) -> str:
        """
        Slogan: Fixes common JSON formatting issues, such as missing quotes.
        Parameters:
            json_str (str): The potentially malformed JSON string.
        Returns:
            str: The corrected JSON string.
        """
        # Fix missing quotes for the function name.
        # This regex finds patterns like: "name": get_current_weather and adds quotes around get_current_weather.
        fixed = re.sub(r'("name":\s*)([a-zA-Z0-9_]+)', r'\1"\2"', json_str)
        return fixed

    def _execute_tool_call(self, tool_call_data: dict) -> str:
        """
        Slogan: Executes a tool call extracted from a tool call data structure.
        Parameters:
            tool_call_data (dict): Contains tool call information with keys 'name' and 'parameters' or 'arguments'.
        Returns:
            str: The result from executing the tool function.
        """
        # Support both formats: {"name": ..., "parameters": ...} and {"function": {"name": ..., "arguments": ...}}
        if "name" in tool_call_data and "parameters" in tool_call_data:
            tool_name = tool_call_data.get("name")
            arguments = tool_call_data.get("parameters", {})
        elif "function" in tool_call_data:
            function_data = tool_call_data.get("function", {})
            tool_name = function_data.get("name")
            arguments = function_data.get("arguments", {})
        else:
            print("[Agent] Tool call data not recognized.")
            return ""

        print("\n[Agent] Detected a tool call for tool:", tool_name)
        # Execute the tool function (using the global registry or a local mapping)
        tool_call = {"tool": tool_name, "args": arguments}
        result = execute_tool_function(tool_call)
        print(f"[Agent] Tool result: {result}")
        # Append the tool response to the conversation history
        self.payload["messages"].append({"role": "tool", "content": result})
        return result

    def process_assistant_response(self, assistant_response: str) -> str:
        """
        Slogan: Processes the assistant response, detects tool calls, and executes them.
        Parameters:
            assistant_response (str): The raw assistant response.
        Returns:
            str: The processed response (tool output if executed, otherwise the original text).
        """
        # Try to extract and process a tool call from the assistant's response
        try:
            data = json.loads(assistant_response)
            message = data.get("message", {})
            # Check for explicit tool_calls list first
            if "tool_calls" in message and isinstance(message["tool_calls"], list) and message["tool_calls"]:
                return self._execute_tool_call(message["tool_calls"][0])
            # Otherwise, try to extract a tool call from the text
            json_substring = extract_json_from_text(message.get("content", ""))
            if json_substring:
                fixed_json = fix_invalid_json(json_substring)
                tool_call_data = json.loads(fixed_json)
                return self._execute_tool_call(tool_call_data)
            return message.get("content", "")
        except json.JSONDecodeError:
            return assistant_response
            
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
                assistant_response = self.client.send_payload(self.payload)
                # Process assistant response for potential tool calls.
                processed_response = self.process_assistant_response(assistant_response)
                # Append assistant (or tool) response.
                self.payload["messages"].append({"role": "assistant", "content": processed_response})

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
    chat_session = OllamaChat(client, config["prompts_file"], config["system"])
    chat_session.run()

if __name__ == "__main__":
    main()
