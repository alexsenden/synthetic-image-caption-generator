"""Dataset loader for reading caption prompts from text files."""

from pathlib import Path
from typing import List


def load_caption_dataset(dataset_dir: Path) -> List[str]:
    """
    Load all prompts from .txt files in the dataset directory.

    Args:
        dataset_dir: Path to directory containing .txt files

    Returns:
        List of all prompts (one prompt per line from all files)
    """
    prompts = []

    # Find all .txt files in the directory
    txt_files = list(dataset_dir.glob("*.txt"))

    if not txt_files:
        return prompts

    # Read prompts from each file
    for txt_file in txt_files:
        try:
            with txt_file.open("r", encoding="utf-8") as f:
                for line in f:
                    # Strip whitespace and skip empty lines
                    line = line.strip()
                    if line:
                        prompts.append(line)
        except Exception as e:
            print(f"Warning: Error reading {txt_file}: {e}")
            continue

    return prompts
