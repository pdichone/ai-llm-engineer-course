<!-- @format -->

# From : https://github.com/ollama/ollama/blob/main/README.md#quickstart

ollama create mario -f ./Modelfile
ollama run mario

> > > hi
> > > Hello! It's your friend Mario.

# Pull a model

ollama pull llama3.2

# Remove a model

ollama rm llama3.2

# multimodal modals -- this will load the llava model ~4.7GB

ollama run llava "What's in this image? /Users/jmorgan/Desktop/smile.png"
The image features a yellow smiley face, which is likely the central focus of the picture.

# Show model info

ollama show llama3.2

# List models on your computer

ollama list

# List which models are currently loaded

ollama ps

# Stop a model which is currently running

ollama stop llama3.2

=====================

# Open WebUI - Installation: https://github.com/open-webui/open-webui?tab=readme-ov-file#-also-check-out-open-webui-community

# Install -- show them this, but we will use msty.app for simplicity

pip install open-webui

=======================

# REST API

ollama has REST API for running and managing models:

# Generate a respnse (this will stream each word...):

curl http://localhost:11434/api/generate -d '{
"model": "llama3.2",
"prompt":"Why is the sky blue?"
}'

# Add stream: false to just get the result:

curl http://localhost:11434/api/generate -d '{
"model": "llama3.2",
"prompt":"tell me a fun fact about Portugal",
"stream": false
}'

# Chat with a model

curl http://localhost:11434/api/chat -d '{
"model": "llama3.2",
"messages": [
{ "role": "user", "content": "tell me a fun fact about Mozambique" }
],
"stream":false
}'

# Request JSON Mode:

curl http://localhost:11434/api/generate -d '{
"model": "llama3.2",
"prompt": "What color is the sky at different times of the day? Respond using JSON",
"format": "json",
"stream": false
}'

# More REST API Documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
