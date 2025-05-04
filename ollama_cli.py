#!/usr/bin/env python3
"""
Ollama CLI Agent with Web Research Tool
- Loads system prompt and tool definitions from prompts.yaml
- Simple command-line interface to a locally hosted LLM
- Supports web research (search + summarization) via self-hosted SearxNG
- Persistent input history for recall
- Configurable model, host, system prompt, and debug logging
"""

import argparse
import inspect
import json
import os
import sys
import logging
import requests
import yaml
from bs4 import BeautifulSoup
from typing import Optional

from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings

# --------------------------
# Defaults and Constants
# --------------------------
DEFAULT_HOST = "localhost:11434"
DEFAULT_MODEL = "gemma3:1b"
DEFAULT_SYSTEM = "You are a helpful assistant."
DEFAULT_CLI_HISTORY_FILE = os.path.expanduser("~/.ollama_cli_prompt_history_file")

# --------------------------
# Load configuration from prompts.yaml, env, and CLI
# --------------------------
def merge_config(args: argparse.Namespace) -> dict:
    """
    Slogan: Merge configuration from CLI args, environment, and prompts.yaml.
    Returns:
        dict with keys host, model, system, tool_definitions, prompts_file.
    """
    # 1) Load YAML if specified or found
    yaml_config = {}
    prompts_file = args.prompts or ("prompts.yaml" if os.path.exists("prompts.yaml") else None)
    if prompts_file:
        try:
            with open(prompts_file, "r") as f:
                yaml_config = yaml.safe_load(f) or {}
        except Exception as e:
            logging.warning(f"[WARNING merge_config]: could not load '{prompts_file}': {e}")

    # 2) Extract from YAML
    yaml_model = yaml_config.get("model")
    yaml_system = yaml_config.get("system")
    yaml_tools  = yaml_config.get("tool_definitions", [])

    # 3) Environment overrides
    env_host  = os.getenv("OLLAMA_HOST")
    env_model = os.getenv("OLLAMA_MODEL")

    # 4) Final precedence: CLI > env > YAML > default
    final_host   = args.host  or env_host  or DEFAULT_HOST
    final_model  = args.model or env_model or yaml_model or DEFAULT_MODEL
    final_system = args.system or yaml_system or DEFAULT_SYSTEM

    return {
        "host": final_host,
        "model": final_model,
        "system": final_system,
        "tool_definitions": yaml_tools,
        "prompts_file": prompts_file
    }

# --------------------------
# Tool Registry
# --------------------------
TOOL_REGISTRY = {}

def register_tool(name: str):
    """
    Slogan: Registers a tool function in the global registry.
    """
    def decorator(func):
        TOOL_REGISTRY[name] = func
        return func
    return decorator

def execute_tool_function(tool_call: dict, client: "OllamaClient") -> dict:
    """
    Slogan: Execute a registered tool based on a function-call spec.
    """
    func_data = tool_call.get("function", {})
    name = func_data.get("name", "")
    args = func_data.get("arguments", {})
    fn = TOOL_REGISTRY.get(name)
    if not fn:
        msg = f"Tool '{name}' not registered."
        logging.error(msg)
        raise ValueError(msg)
    try:
        sig = inspect.signature(fn)
        if "client" in sig.parameters:
            args["client"] = client
        result = fn(**args)
        return {"role": "tool", "content": result}
    except Exception as e:
        logging.error("Error executing %s: %s", name, e)
        return {"role": "tool", "content": f"Error executing {name}: {e}"}

# --------------------------
# Tools
# --------------------------

def web_search(query: str, k: int = 5, searx_url: Optional[str] = None) -> str:
    """
    Slogan: Query SearxNG and return the top k results as “[1] title – url – snippet.”
    """
    base = (searx_url or os.getenv("SEARXNG_URL", "http://localhost:8080")).rstrip("/")
    k = max(1, min(int(k), 10))
    params = {"q": query, "format": "json", "language": "en", "safesearch": 1, "categories": "general"}
    try:
        r = requests.get(f"{base}/search", params=params, timeout=8)
        r.raise_for_status()
        items = r.json().get("results", [])[:k]
    except Exception as e:
        logging.error("SearxNG request failed: %s", e)
        raise RuntimeError(f"SearxNG request failed: {e}")
    out = []
    for i, it in enumerate(items, 1):
        title = it.get("title","").strip()
        url   = it.get("url","").strip()
        snippet = it.get("content","").replace("\n"," ").strip()
        out.append(f"[{i}] {title} – {url} – {snippet}")
    return "\n".join(out) if out else "No results found."

def get_webpage_text(url: str) -> str:
    """
    Slogan: Retrieve and clean text from a web page.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=8)
        r.raise_for_status()
    except Exception as e:
        return f"Error fetching {url}: {e}"
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script","style"]):
        tag.extract()
    return soup.get_text(separator=" ", strip=True)

@register_tool("research_query")
def research_query(
    query: str,
    top_k: Optional[int] = 5,
    searx_url: Optional[str] = None,
    client=None
) -> str:
    """
    Slogan: Web research workflow: search → fetch text → summarize → synthesize.
    """
    if client is None:
        return "Internal error: missing LLM client."
    hits = web_search(query=query, k=top_k, searx_url=searx_url).splitlines()
    excerpts = []
    for i, line in enumerate(hits, 1):
        parts = line.split("–")
        url = parts[1].strip() if len(parts)>1 else ""
        text = get_webpage_text(url=url)
        excerpts.append(f"--- Article {i} ({url}) ---\n{text}\n")
    prompt = (
        f"You are an expert research assistant. Query: {query}\n\n"
        + "\n".join(excerpts)
        + "\nPlease summarize each article and synthesize an overall answer."
    )
    return client.send_direct_prompt(prompt)

# --------------------------
# Ollama Client
# --------------------------

class OllamaClient:
    """
    Slogan: Client for interacting with the local Ollama server.
    """
    def __init__(self, host: str, model: str, debug: bool=False):
        self.host = host
        self.model = model
        self.debug = debug
        self.base_url = f"http://{host}"
        self.config = {}  # will be set in main()

    def _url(self, endpoint: str) -> str:
        return f"{self.base_url}{endpoint}"

    def check_connection(self):
        try:
            r = requests.get(self._url("/api/tags"), timeout=3)
            r.raise_for_status()
            logging.info("✅ Connected to Ollama server at %s", self.host)
        except Exception as e:
            logging.error("❌ Cannot reach Ollama server: %s", e)
            sys.exit(1)

    def verify_model(self):
        try:
            r = requests.get(self._url("/api/tags"), timeout=3)
            r.raise_for_status()
            available = [m["name"] for m in r.json().get("models",[])]
            if self.model not in available:
                logging.error("❌ Model '%s' not available. Available: %s", self.model, available)
                sys.exit(1)
            logging.info("✅ Using model: %s", self.model)
        except Exception as e:
            logging.error("❌ Failed fetching models: %s", e)
            sys.exit(1)

    def send_direct_prompt(self, prompt_str: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role":"user","content":prompt_str}],
            "stream": False
        }
        r = requests.post(self._url("/api/chat"), json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _process_tool_calls(self, data: dict) -> list:
        msgs = []
        for call in data.get("message",{}).get("tool_calls",[]):
            msgs.append(execute_tool_function(call, self))
            if self.debug:
                logging.debug("Tool call: %s → %s", call, msgs)
        return msgs

    def send_payload(self, payload: dict) -> list:
        """
        Slogan: Send the conversation payload to /api/chat with streaming,
        printing each partial assistant response as it arrives.
        """
        if self.debug:
            logging.debug("Request payload: %s", payload)

        messages = []
        # Make sure we're explicitly requesting streaming at the HTTP level too
        headers = {"Accept": "text/event-stream"}  

        with requests.post(
            self._url("/api/chat"),
            json=payload,
            stream=True,
            headers=headers,
            timeout=(3.05, None)  # no read timeout
        ) as resp:
            resp.raise_for_status()
            print(f"{self.model}> ", end="", flush=True)
            full = ""

            # chunk_size=1 forces us to see each newline-delimited JSON as soon as it's sent
            for line in resp.iter_lines(chunk_size=1, decode_unicode=True):
                if not line:
                    continue
                # each 'line' should be a complete JSON object
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    # sometimes you get keep-alive pings—ignore them
                    continue

                txt = chunk.get("message", {}).get("content", "") or ""
                full += txt

                # print each piece immediately
                print(txt, end="", flush=True)

                # process any tool calls embedded in this chunk
                tool_msgs = self._process_tool_calls(chunk)
                for tm in tool_msgs:
                    messages.append(tm)

            # final newline after the stream
            print()

        # append the combined assistant message at the end of tool calls
        messages.append({"role": "assistant", "content": full})
        return messages

# --------------------------
# CLI Chat Session
# --------------------------

class OllamaChat:
    """
    Slogan: Manage interactive CLI chat with the LLM.
    """
    def __init__(self, client: OllamaClient, system_prompt: str, history_file: str, debug: bool=False):
        self.client = client
        self.system = system_prompt
        self.history_file = history_file
        self.debug = debug
        # initial payload
        self.payload = {
            "model": client.model,
            "messages": [{"role":"system","content":self.system}],
            "stream": True
        }
        # inject tools from prompts.yaml
        td = client.config.get("tool_definitions", [])
        if td:
            self.payload["tools"] = td

    def _send_payload(self, user_input: str):
        self.payload["messages"].append({"role":"user","content":user_input})
        msgs = self.client.send_payload(self.payload)
        # merge tool + assistant messages into our payload
        if self.debug:
            logging.debug("Payload before merging tool msgs: %s", self.payload)
        for m in msgs:
            # add either the tool output or the assistant’s final content
            if m.get("content"):
                self.payload["messages"].append(m)
        if self.debug:
            logging.debug("Payload after merging tool msgs: %s", self.payload)
        # if the LLM requested a tool, ask the LLM to act on the tool's response(s)
        if self.payload["messages"][-1].get("role") == "tool":
            # this sends the tool response back into the LLM 
            # so it can produce the final natural-language answer
            self.payload["messages"].append({
                "role":"user",
                "content":"Make an observation of the tool responses in the current context, and based on your reasoning, take the next step. If you conclude there are no additional steps, exit your ReAct loop and issue your Final Answer. AVOID entering into recursive calls with tools!"
            })
            # Perform the synthesis call
            msgs = self.client.send_payload(self.payload)

            # Merge the synthesis messages (should be assistant content only)
            if msgs:
                for m in msgs:
                    self.payload["messages"].append(m)
            else:
                # fallback if something went wrong
                self.payload["messages"].append({
                    "role":"assistant",
                    "content":"I could not synthesize a response!"
                })

    def _setup_key_bindings(self):
        kb = KeyBindings()
        @kb.add("c-c")
        def _(event):
            print("\nExiting...")
            event.app.exit()
        return kb

    def _load_history(self):
        hist = InMemoryHistory()
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file) as f:
                    for line in f:
                        hist.append_string(line.rstrip("\n"))
            except Exception as e:
                logging.error("Load history failed: %s", e)
        return hist

    def _save_history(self, history: InMemoryHistory):
        try:
            last = history.get_strings()[-250:]
            with open(self.history_file,"w") as f:
                f.write("\n".join(last))
        except Exception as e:
            logging.error("Save history failed: %s", e)

    def run(self):
        kb = self._setup_key_bindings()
        history = self._load_history()
        print("Welcome to Ollama CLI Agent. Type 'exit' or 'quit' to end.")
        while True:
            try:
                inp = prompt("Your query?> ", history=history,
                             auto_suggest=AutoSuggestFromHistory(),
                             key_bindings=kb)
                if inp.strip().lower() in {"exit","quit"}:
                    break
                if not inp.strip():
                    continue
                print("\nStarting inference...\n")
                self._send_payload(inp)
            except (EOFError, KeyboardInterrupt):
                break
        print("\nGoodbye!")
        self._save_history(history)

# --------------------------
# Argument Parsing & Main
# --------------------------

def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ollama CLI Agent – Local LLM + Web Research")
    p.add_argument("--host",         help="Ollama host:port", default=None)
    p.add_argument("--model",        help="Model name",       default=None)
    p.add_argument("--system",       help="System prompt",    default=None)
    p.add_argument("--prompts",      help="prompts.yaml file",default=None)
    p.add_argument("--debug",        action="store_true",   help="Enable debug logging")
    p.add_argument("--history-file", help="CLI history file", default=DEFAULT_CLI_HISTORY_FILE)
    return p.parse_args()

def main():
    args   = parse_arguments()
    config = merge_config(args)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(levelname)s] %(message)s"
    )

    client = OllamaClient(host=config["host"], model=config["model"], debug=args.debug)
    client.config = config

    client.check_connection()
    client.verify_model()

    chat = OllamaChat(
        client=client,
        system_prompt=config["system"],
        history_file=args.history_file,
        debug=args.debug
    )
    chat.run()

if __name__ == "__main__":
    main()
