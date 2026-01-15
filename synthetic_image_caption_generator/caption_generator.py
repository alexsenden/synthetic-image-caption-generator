"""Caption generator using Qwen2.5-32B-Instruct model."""

import random
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class CaptionGenerator:
    """Generate captions similar to example prompts using Qwen2.5-32B-Instruct."""

    def __init__(
        self,
        model: str = "qwen2.5-32b",
        temperature: float = 0.7,
        max_length: int = 256,
    ):
        """
        Initialize the caption generator.

        Args:
            model: Qwen model to use (e.g., "qwen2.5-32b", "qwen3-14b")
            temperature: Temperature for text generation (higher = more random)
            max_length: Maximum length of generated text
        """
        self.temperature = temperature
        self.max_length = max_length

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

        if model.lower() not in model_map:
            raise ValueError(
                f"Unknown model: {model}. Supported models: {', '.join(model_map.keys())}"
            )

        model_name = model_map[model.lower()]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()

    def generate(
        self, prompts: List[str], num_examples: int = 5, object_info: str = None
    ) -> str:
        """
        Generate a new caption similar to the example prompts.

        Args:
            prompts: List of all available prompts from the dataset
            num_examples: Number of example prompts to show the model
            object_info: Optional information about objects/content in the image

        Returns:
            Generated caption as a string
        """
        # Randomly sample example prompts
        examples = random.sample(prompts, min(num_examples, len(prompts)))

        # Construct the prompt for the model
        system_message = """You are a caption generation assistant. Your task is to generate image captions that are similar in style and structure to the examples provided. Study the examples carefully and create a new caption that matches their pattern, tone, and level of detail."""

        # Build examples section
        examples_text = "\n".join([f"- {example}" for example in examples])

        # Build user message with optional object info
        if object_info:
            user_message = f"""Here are some example image captions:

{examples_text}

I need to create a caption for an image where {object_info}.

Based on the example captions above, generate a new image caption that follows the same style, structure, and level of detail, while incorporating the information about the image content. Output only the caption itself, without any additional explanation or formatting."""
        else:
            user_message = f"""Here are some example image captions:

{examples_text}

Based on these examples, generate a new image caption that follows the same style, structure, and level of detail. Output only the caption itself, without any additional explanation or formatting."""

        # Format using Qwen chat template
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode and extract just the generated text
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Clean up the output
        generated_caption = generated_text.strip()

        return generated_caption
