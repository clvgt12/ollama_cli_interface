import argparse
import requests
import json
import datetime
import os
import pytz
import yaml
import re
from smolagents import CodeAgent, HfApiModel, tool
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings

# Default values
DEFAULT_HOST = "localhost:11434"
DEFAULT_MODEL = "mistral:latest"

# Set up argument parsing
parser = argparse.ArgumentParser(description="Ollama AI Agent CLI - Interact with an AI agent using a local LLM.")
parser.add_argument("--model", type=str, default=os.getenv("MODEL_NAME", DEFAULT_MODEL), help=f"Model name to use (default: {DEFAULT_MODEL})")
parser.add_argument("--host", type=str, default=os.getenv("OLLAMA_HOST", DEFAULT_HOST), help=f"Ollama server host (default: {DEFAULT_HOST})")
parser.add_argument("--debug", action="store_true", help="Enable debug mode to print raw responses")
args = parser.parse_args()

# Assign parsed arguments
OLLAMA_HOST = args.host
MODEL_NAME = args.model
DEBUG_MODE = args.debug

# Connection Check: Ensure Ollama server is reachable
def check_ollama_connection():
    OLLAMA_API_URL = f"http://{OLLAMA_HOST}/api/tags"
    try:
        response = requests.get(OLLAMA_API_URL, timeout=3)
        response.raise_for_status()
        print("✅ Successfully connected to Ollama server.")
    except requests.RequestException:
        print(f"❌ Error: Unable to reach Ollama server at {OLLAMA_HOST}.")
        exit(1)

check_ollama_connection()

# -------------------- Custom Tool: Calculator -------------------- #
@tool
def calc(expression: str) -> str:
    """
    Evaluates an arithmetic expression.

    Args:
        expression (str): A mathematical expression (e.g., "2+3*4").
    """
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"

# -------------------- Custom Tool: Timezone ---------------------- #
@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """
    Retrieves the current time for a specified timezone.

    Args:
        timezone (str): The timezone to retrieve the time for (e.g., 'America/New_York').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

# -------------------- Ollama API Interaction -------------------- #
def query_ollama(prompt_text, chat_memory):
    OLLAMA_API_URL = f"http://{OLLAMA_HOST}/api/generate"
    available_tools = [
        {
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Evaluates a mathematical expression and returns the result.",
                "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "The arithmetic expression to evaluate"}}, "required": ["expression"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time_in_timezone",
                "description": "Retrieves the current time in a specified timezone.",
                "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "The timezone, e.g., 'Europe/Amsterdam'."}}, "required": ["timezone"]}
            }
        }
    ]

    # Construct memory-aware prompt
    memory_context = "\n".join(chat_memory[-6:])  # Keep last 6 interactions
    formatted_prompt = f"[AVAILABLE_TOOLS] {json.dumps(available_tools)} [/AVAILABLE_TOOLS]\n"
    formatted_prompt += f"[MEMORY] {memory_context} [/MEMORY]\n"
    formatted_prompt += f"[INST] {prompt_text} [/INST]"

    payload = {"model": MODEL_NAME, "prompt": formatted_prompt, "stream": True, "raw": True}

    try:
        with requests.post(OLLAMA_API_URL, json=payload, stream=True) as response:
            response.raise_for_status()
            print(f"\n{MODEL_NAME}: ", end="", flush=True)
            full_response = ""

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if DEBUG_MODE:
                            print("\nDEBUG - Raw Response:", json.dumps(data, indent=2), "\n")
                        response_text = data.get("response", "")
                        full_response += response_text
                        print(response_text, end="", flush=True)
                    except json.JSONDecodeError:
                        continue

            chat_memory.append(f"user: {prompt_text}")
            chat_memory.append(f"assistant: {full_response}")

    except requests.RequestException as e:
        print(f"\nError: Unable to reach Ollama server. {e}")

# -------------------- CLI Main Loop -------------------- #
def main():
    history = InMemoryHistory()
    print(f"Connected to Ollama at {OLLAMA_HOST}")
    print(f"Using model: {MODEL_NAME}")
    if DEBUG_MODE:
        print("🔍 Debug mode enabled - Printing raw responses")
    print("Welcome to the AI CLI Agent - Type your prompt and press Enter.")
    print("Type 'exit' or 'quit' to end the session.")

    # Conversation memory (keeps track of previous exchanges)
    chat_memory = []

    while True:
        try:
            user_input = prompt("\nAI> ", history=history, auto_suggest=AutoSuggestFromHistory())
            if user_input.lower() in ["exit", "quit"]:
                break
            query_ollama(user_input, chat_memory)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
