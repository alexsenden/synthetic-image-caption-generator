# Synthetic Image Caption Generator

A tool that uses Qwen to generate image captions similar to those in a given dataset. The model learns from example prompts and generates new captions that match their style and structure.

This tool is designed to be used for deep generative dataset augmentation.

## Features

-   Uses Qwen from HuggingFace🤗 Transformers
-   Supports few-shot learning with configurable number of examples
-   Reads prompts from .txt files in a dataset directory
-   Flexible CLI with multiple configuration options
-   Can generate multiple captions in one run
-   Output to file or stdout

## Installation

To install from PyPI:

```bash
pip install synthetic-image-caption-generator
```

To install locally:

```bash
pip install -e .
```

Or install dependencies manually:

```bash
pip install transformers>=4.30.0 torch>=2.0.0 accelerate>=0.20.0
```

## Usage

### Basic Usage

Generate a caption using 5 example prompts from the dataset:

```bash
generate-captions /path/to/dataset
```

### Specify Number of Examples

Use a different number of example prompts (e.g., 10):

```bash
generate-captions /path/to/dataset --num-examples 10
```

### Choose a Model

Select a different Qwen model (e.g., smaller or larger):

```bash
# Use the smaller 7B model
generate-captions /path/to/dataset --model qwen2.5-7b

# Use the larger 72B model
generate-captions /path/to/dataset --model qwen2.5-72b

# Use Qwen3 14B
generate-captions /path/to/dataset --model qwen3-14b
```

### Generate Multiple Captions

Generate 5 captions:

```bash
generate-captions /path/to/dataset --num-generate 5
```

### Specify Object Information

Include information about what's in the image:

```bash
generate-captions /path/to/dataset --object-info "the image contains 3 elephants"
```

```bash
generate-captions /path/to/dataset --object-info "the main crop in the field is soybean"
```

### Save to File

Save generated captions to a file:

```bash
generate-captions /path/to/dataset --num-generate 10 --output generated_captions.txt
```

### Advanced Options

```bash
generate-captions /path/to/dataset \
  --model qwen2.5-14b \
  --num-examples 8 \
  --num-generate 5 \
  --temperature 0.8 \
  --max-length 300 \
  --object-info "a cityscape with tall buildings" \
  --output captions.txt
```

## Command-Line Arguments

-   `dataset_dir` (required): Path to directory containing .txt files with caption prompts
-   `--model`: Qwen model to use (default: qwen2.5-32b). Options: qwen2.5-0.5b, qwen2.5-1.5b, qwen2.5-3b, qwen2.5-7b, qwen2.5-14b, qwen2.5-32b, qwen2.5-72b, qwen3-14b, qwen3-32b
-   `--num-examples`: Number of example prompts to provide to the model (default: 5)
-   `--num-generate`: Number of captions to generate (default: 1)
-   `--output`: Output file to save generated captions (optional, prints to stdout if not specified)
-   `--temperature`: Temperature for text generation (default: 0.7, higher = more creative)
-   `--max-length`: Maximum length of generated caption (default: 256)
-   `--object-info`: Information about objects/content in the image (e.g., "the image contains 3 elephants")

## Dataset Format

The dataset directory should contain `.txt` files, each with one or more prompts. Each prompt should be on its own line.

Example dataset structure:

```
dataset/
├── captions1.txt
├── captions2.txt
└── captions3.txt
```

Example content of `captions1.txt`:

```
A serene landscape with mountains in the background and a lake in the foreground
An urban street scene with people walking and cars passing by
A close-up portrait of a person smiling at the camera
```

## Requirements

-   Python >= 3.8
-   CUDA-capable GPU recommended
-   Sufficient GPU memory depending on chosen model:
    -   Small models (0.5B-3B): 4-8GB VRAM
    -   Medium models (7B-14B): 12-24GB VRAM
    -   Large models (32B): 24-40GB VRAM
    -   Extra large models (72B): 40GB+ VRAM or quantization required

## How It Works

1. The script reads all prompts from .txt files in the specified directory
2. It randomly samples a specified number of prompts as examples
3. These examples are formatted into a few-shot prompt for Qwen2.5-32B-Instruct
4. The model generates a new caption that matches the style and structure of the examples
5. The generated caption(s) are output to stdout or saved to a file

## License

MIT
