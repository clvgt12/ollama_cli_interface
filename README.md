# Ollama Agent CLI tool

![](./images/robot_at_computer_dalle3.jpg "Source: Dall-E 3: Image of a AI robot, typing at a computer")

## Introduction
In February 2025 I participated in the Hugging Face AI Agent [course](./images/hugging_face_ai_agent_certificate_Feb_2025.jpg "Hugging Face AI Agent certificate of completion issued to Chris Vitalos February 2025"). Part of the course was a coding challenge to create a Python based AI agent using Hugging Face Spaces. The free virtual CPU provided by Spaces quickly became overloaded, which dramatically slowed down response times.

I got quickly frustrated by the lack of progress, and decided to strike out on my own to use locally served LLM via an Ollama inference server, with a CLI to accept the prompts and display the response.

## Use Case
1. As a user I want to use a simple command line interface to issue prompts to an large language model.
2. As a AI agent learner, I want to create a AI agent to help me learn agentic AI architectures, by have the agent interact with a LLM and invoke tools that I create for specific queries.

## Installation Instructions
### Prerequisites
- Python 3.12 or later
- Ollama, installed on your host computer, along with mistral v0.3 LLM to support AI agent use of tools
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
$ python3 ./ollama_agent_cli.py 
```
## Usage

You will see a command line prompt starting with the characters "AI>"  Here you will prompt the LLM with your queries and see responses.  If you ask the agent the following related queries, it is designed to invoke internal tools to produce the result.
1. Simple math equations (add, subtract, multiply, divide)
2. Current time in a city you specify 
