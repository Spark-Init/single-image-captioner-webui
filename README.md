# Captioning and tagging web interface

A minimal web interface for querying various image captioning and tagging models. Built for personal testing and single-image usage.

## Models

- Image Captioning: Meta-Llama 3 with SigLIP (CLIP)
- JoyTag: https://huggingface.co/fancyfeast/joytag 
- WD Tagger v3: https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3

## Requirements

A CUDA-capable GPU with enough VRAM to run Llama 3 models.

## Installation & Usage

1. Run `run.bat` to install requirements and start the server.
3. Access via `http://localhost:5000`