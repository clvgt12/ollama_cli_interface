import argparse
import requests
import json
import datetime
import os
import pytz
import yaml
import re
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from dotenv import load_dotenv

load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool

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

# -------------------- Custom Tool: Calculator -------------------- #
def calc(expression: str) -> str:
    """A simple calculator tool that evaluates arithmetic expressions.
        Args:
            expression: A string containing a mathematical expression (e.g., "2+3*4").
    """
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"

calc_tool = FunctionTool.from_defaults(fn=calc)

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

# -------------------- Custom Tool: Timezone ---------------------- #
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
        Args:
            timezone: A string representing a valid timezone (e.g., 'Europe/Amsterdam').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return local_time
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

timezone_tool = FunctionTool.from_defaults(fn=get_current_time_in_timezone)

# CLI Main Loop
def main():

    print(f"Connected to Ollama at {OLLAMA_HOST}")
    print(f"Using model: {MODEL_NAME}")
    if DEBUG_MODE:
        print("🔍 Debug mode enabled - Printing raw responses")
    print("Welcome to the AI CLI Agent - Type your prompt and press Enter.")
    print("Type 'exit' or 'quit' to end the session.")

    history = InMemoryHistory()
    llm = Ollama(model=MODEL_NAME, request_timeout=120.0)
    agent = ReActAgent.from_tools([add_tool, multiply_tool, timezone_tool], llm=llm)
    chat_memory = []
    while True:
        try:
            user_input = prompt("\nYour prompt?> ", history=history, auto_suggest=AutoSuggestFromHistory())
            if user_input.lower() in ["exit", "quit"]:
                break
            response = agent.chat(user_input)
            print(f"{MODEL_NAME}> {response}")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
