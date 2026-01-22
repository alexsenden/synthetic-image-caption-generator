"""Command-line interface for the synthetic image caption generator."""

import argparse
import sys
from pathlib import Path

from .dataset_loader import load_caption_dataset
from .caption_generator import CaptionGenerator


def download_model_cli():
    """Download a Qwen model to the Hugging Face cache."""
    parser = argparse.ArgumentParser(
        description="Download a Qwen model to the Hugging Face cache for offline use"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-32b",
        choices=[
            "qwen2.5-0.5b",
            "qwen2.5-1.5b",
            "qwen2.5-3b",
            "qwen2.5-7b",
            "qwen2.5-14b",
            "qwen2.5-32b",
            "qwen2.5-72b",
            "qwen3-14b",
            "qwen3-32b",
        ],
        help="Model to download (default: qwen2.5-32b)",
    )

    args = parser.parse_args()

    # Map model names to Hugging Face model IDs
    model_map = {
        "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
        "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
        "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
        "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
        "qwen3-14b": "Qwen/Qwen3-14B-Instruct",
        "qwen3-32b": "Qwen/Qwen3-32B-Instruct",
    }

    model_name = model_map[args.model]

    print(f"Downloading {model_name} to Hugging Face cache...", file=sys.stderr)
    print("This may take a while depending on the model size.", file=sys.stderr)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Download tokenizer
        print(f"\nDownloading tokenizer...", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Tokenizer downloaded successfully", file=sys.stderr)

        # Download model
        print(f"\nDownloading model weights...", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        print(f"✓ Model downloaded successfully", file=sys.stderr)

        print(f"\n✓ {model_name} is now cached and ready for offline use!", file=sys.stderr)

    except Exception as e:
        print(f"\n✗ Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic image captions using Qwen2.5-32B-Instruct"
    )

    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to directory containing .txt files with caption prompts",
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of example prompts to provide to the model (default: 5)",
    )

    parser.add_argument(
        "--num-generate",
        type=int,
        default=1,
        help="Number of captions to generate (default: 1)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save generated captions (optional, prints to stdout if not specified)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for text generation (default: 0.7)",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum length of generated caption (default: 256)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-32b",
        choices=[
            "qwen2.5-0.5b",
            "qwen2.5-1.5b",
            "qwen2.5-3b",
            "qwen2.5-7b",
            "qwen2.5-14b",
            "qwen2.5-32b",
            "qwen2.5-72b",
            "qwen3-14b",
            "qwen3-32b",
        ],
        help="Model to use for caption generation (default: qwen2.5-32b)",
    )

    parser.add_argument(
        "--object-info",
        type=str,
        default=None,
        help="Information about objects/content in the image (e.g., 'the image contains 3 elephants')",
    )

    args = parser.parse_args()

    # Validate dataset directory
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(
            f"Error: Dataset directory '{args.dataset_dir}' does not exist",
            file=sys.stderr,
        )
        sys.exit(1)

    if not dataset_path.is_dir():
        print(f"Error: '{args.dataset_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)

    # Load dataset
    print(f"Loading prompts from {dataset_path}...", file=sys.stderr)
    prompts = load_caption_dataset(dataset_path)

    if len(prompts) == 0:
        print("Error: No prompts found in dataset directory", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(prompts)} prompts from dataset", file=sys.stderr)

    if args.num_examples > len(prompts):
        print(
            f"Warning: Requested {args.num_examples} examples but only {len(prompts)} available. Using all available prompts.",
            file=sys.stderr,
        )
        args.num_examples = len(prompts)

    # Initialize generator
    print("Loading model...", file=sys.stderr)
    generator = CaptionGenerator(
        model=args.model, temperature=args.temperature, max_length=args.max_length
    )

    # Generate captions
    print(
        f"Generating {args.num_generate} caption(s) using {args.num_examples} example(s)...",
        file=sys.stderr,
    )
    if args.object_info:
        print(f"Object info: {args.object_info}", file=sys.stderr)
    generated_captions = []

    for i in range(args.num_generate):
        caption = generator.generate(
            prompts, num_examples=args.num_examples, object_info=args.object_info
        )
        generated_captions.append(caption)
        print(f"Generated caption {i+1}/{args.num_generate}", file=sys.stderr)

    # Output results
    if args.output:
        output_path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as f:
            for caption in generated_captions:
                f.write(caption + "\n")
        print(
            f"Saved {len(generated_captions)} caption(s) to {args.output}",
            file=sys.stderr,
        )
    else:
        print("\n--- Generated Captions ---", file=sys.stderr)
        for i, caption in enumerate(generated_captions, 1):
            print(f"\n{i}. {caption}")

    print("\nDone!", file=sys.stderr)


if __name__ == "__main__":
    main()
