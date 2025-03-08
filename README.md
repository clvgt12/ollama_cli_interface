# Ollama Agent CLI tool

![Llama Shepherd](./images/robot-shepherd-flock-of-llamas.jpg "Source: Dall-E 3: Image of a AI robot, typing at a computer")

## Introduction
In February 2025 I participated in the Hugging Face AI Agent [course](./images/hugging_face_ai_agent_certificate_Feb_2025.jpg "Hugging Face AI Agent certificate of completion issued to Chris Vitalos February 2025"). Part of the course was a coding challenge to create a Python based AI agent using Hugging Face Spaces. 

I decided to strike out on my own to use locally served small language models via an [Ollama](https://ollama.ai) inference server running on my laptop, with a CLI to accept the prompts and display the response.

## Use Case
1. As a user I want to use a simple command line interface to issue prompts to the language model.
2. As a AI agent learner, I want to create a AI agent to help me learn agentic AI architectures, by have the agent interact with a language model and invoke tools that I create for specific queries.

## Installation Instructions
### Prerequisites
- Python 3.12 or later
- Ollama, installed on your host computer, along with mistral v0.3 LLM to support AI agent use of tools
- An [OpenWeatherMap.org](https://home.openweathermap.org/api_keys) API key to enable the agent to provide current weather given a specified location
- An [Geoapify.com](https://www.geoapify.com/api/) API key to enable the agent to obtain an exact geolocation of the provided location.
### Python and Required Modules Installation
#### Windows
1. Download and install Python from [python.org](https://www.python.org/downloads/windows/).
2. Install required modules via pip
```
$ pip install -r requirements.txt
```
#### Linux/MacOS
1. Python is typically pre-installed on Linux and macOS. If not, install Python using your distribution's package manager (Linux) or download from [python.org](https://www.python.org/downloads/macos/) (macOS).
2. Install Streamlit and other required modules:
```
$ pip3 install -r requirements.txt
```
## Running the Application
#### Windows, Linux and MacOS
From the command line, run:
```
$ ollama pull mistral:latest
$ ollama serve
$ echo "Your_OpenWeatherMaps_API_key" > .env
$ echo "Your_Geoapify_API_Key" >> .env
$ python3 ./ollama_cli.py --prompts=PROMPTS_YAML_FILENAME
```
## Prompts.yaml files

The system prompts can be customized by providing different yaml files to the Ollama served language models. To enable an AI agent that can gather current weather conditions given a specified location, invoke `ollama_cli.py` with the option `--prompts="tools-prompts.yaml"`

## Usage

You will see a command line prompt starting with the characters "Your query>"  Here you will prompt the LLM with your queries and see responses.  

If you provided the agent with the `tools-prompts.yaml` file, and ask the agent to provide current weather conditions given a location, i.e., city, state and country, it will summarize information it gathers from [OpenWeatherMap.org](https://openweathermap.org) to synthesize a response.
