#!/usr/bin/env python3
"""
Example script showing how to use the caption generator programmatically.
"""

from pathlib import Path
from synthetic_image_caption_generator.dataset_loader import load_caption_dataset
from synthetic_image_caption_generator.caption_generator import CaptionGenerator


def main():
    # Load dataset
    dataset_path = Path("example_dataset")
    prompts = load_caption_dataset(dataset_path)

    print(f"Loaded {len(prompts)} prompts from dataset\n")
    print("Example prompts:")
    for i, prompt in enumerate(prompts[:3], 1):
        print(f"{i}. {prompt}")
    print()

    # Initialize generator
    print("Loading model...")
    # You can specify different models here
    generator = CaptionGenerator(model="qwen2.5-0.5b", temperature=0.7, max_length=256)

    # Generate a caption
    print("\nGenerating new caption...")
    caption = generator.generate(prompts, num_examples=5)

    print("\nGenerated caption:")
    print(f"→ {caption}")

    # Generate with object information
    print("\nGenerating caption with object info...")
    caption_with_info = generator.generate(
        prompts, num_examples=5, object_info="a dog playing with a ball in a park"
    )

    print("\nGenerated caption with object info:")
    print(f"→ {caption_with_info}")


if __name__ == "__main__":
    main()
