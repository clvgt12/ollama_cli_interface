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

# -------------------- Custom Tool: Calculator -------------------- #
@tool
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

# -------------------- Custom Tool: Timezone ---------------------- #
@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'Europe/Amsterdam').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

# -------------------- Ollama API Interaction -------------------- #
def query_ollama(prompt_text):
    """
    Sends a prompt to the Ollama inference server, correctly formatting it for raw mode function calling.
    """
    OLLAMA_API_URL = f"http://{OLLAMA_HOST}/api/generate"
    available_tools = [
        {
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Evaluates a mathematical expression and returns the result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "The arithmetic expression to evaluate"}
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time_in_timezone",
                "description": "Retrieves the current time in a specified timezone.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string", "description": "The timezone, e.g., 'Europe/Amsterdam'."}
                    },
                    "required": ["timezone"]
                }
            }
        }
    ]

    formatted_prompt = f"[AVAILABLE_TOOLS] {json.dumps(available_tools)} [/AVAILABLE_TOOLS]\n[INST] {prompt_text} [/INST]"

    payload = {
        "model": MODEL_NAME,
        "prompt": formatted_prompt,
        "stream": True,
        "raw": True  # Enable raw mode for function calling
    }

    try:
        with requests.post(OLLAMA_API_URL, json=payload, stream=True) as response:
            response.raise_for_status()
            print(f"\n{MODEL_NAME}: ", end="", flush=True)

            full_response = ""  # Store accumulated response
            tool_call_detected = False  # Track if a tool call occurs

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)

                        # Debug Mode: Print the raw response
                        if DEBUG_MODE:
                            print("\nDEBUG - Raw Response:", json.dumps(data, indent=2), "\n")

                        response_text = data.get("response", "")
                        full_response += response_text

                        # If a tool call is detected, process it but DON'T print it as part of LLM response
                        if "[" in response_text and "{" in response_text:
                            tool_call_detected = True
                        elif not tool_call_detected:
                            print(response_text, end="", flush=True)  # Stream normal LLM output

                    except json.JSONDecodeError:
                        continue

            # Extract and process JSON tool calls
            tool_call_match = re.search(r"\[(\{.*?\})\]", full_response, re.DOTALL)
            if tool_call_match:
                tool_calls_json = "[" + tool_call_match.group(1) + "]"
                try:
                    tool_calls = json.loads(tool_calls_json)

                    for call in tool_calls:
                        tool_name = call.get("name")
                        tool_args = call.get("arguments", {})

                        if tool_name == "get_current_time_in_timezone":
                            timezone = tool_args.get("timezone", "UTC")
                            result = get_current_time_in_timezone(timezone)
                            print("\n" + result, flush=True)  # ✅ Print only tool output

                        elif tool_name == "calc":
                            expression = tool_args.get("expression", "")
                            result = calc(expression)
                            print("\n" + result, flush=True)  # ✅ Print only tool output

                except json.JSONDecodeError:
                    print("\n⚠️ ERROR: Failed to parse tool calls JSON.\n")

    except requests.RequestException as e:
        print(f"\nError: Unable to reach Ollama server. {e}")

# -------------------- AI Agent Definition -------------------- #
model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id=f"http://{OLLAMA_HOST}",
    custom_role_conversions=None,
)

with open("prompts.yaml", "r") as file:
    prompt_templates = yaml.safe_load(file)

agent = CodeAgent(
    model=model,
    tools=[calc, get_current_time_in_timezone],  # Add tools
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name="CLI_Agent",
    description="A CLI-based AI agent using Ollama LLM",
    prompt_templates=prompt_templates,
)

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
