"""
VLM-Streaming
StreamVLM: A class for visual language model text generation with streaming output.

This module provides a streaming interface to visual language models like Qwen2.5-VL,
allowing token-by-token generation for real-time applications with image inputs.
"""
from threading import Thread
from typing import Dict, Generator
import os
import torch
import base64
from io import BytesIO
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    GenerationConfig,
    TextIteratorStreamer,
)
from PIL import Image


class StreamVLM:
    """Streaming visual language model wrapper."""
    
    def __init__(self, config) -> None:
        """
        Initialize model with configuration parameters.
        
        Args:
            config: Dictionary with configuration parameters
        """
        self.config = config
        self.processor = None
        self.model = None
        self.device = config.get("llm_device", "cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Set default parameters from config or use defaults
        self.model_name = config.get("llm_checkpoint", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.max_tokens = config.get("llm_max_tokens", 1000)
        self.temperature = config.get("llm_temperature", 0.2)
        self.top_p = config.get("llm_top_p", 0.95)
        self.top_k = config.get("llm_top_k", 40)
        
        # Image processor settings
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 1280 * 28 * 28

    def load(self):
        """
        Load the model and processor.
        
        The model is loaded onto the specified device.
        """
        print(f"Loading VLM from {self.model_name}...")
        
        # Load the model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        
        # Load the processor
        processor_name = self.config.get("llm_checkpoint", "Qwen/Qwen2-VL-7B-Instruct")
        self.processor = AutoProcessor.from_pretrained(
            processor_name,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            use_fast=True
        )
        
        print(f"VLM loaded successfully on {self.device}")

    def decode_base64_to_image(self, base64_string):
        """
        Decode a base64 string into a PIL Image.
        
        Args:
            base64_string: Base64-encoded image string
            
        Returns:
            PIL.Image.Image: The decoded image in RGB format
        """
        # Decode the base64 string back to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Create a BytesIO object from the decoded bytes
        image_buffer = BytesIO(image_bytes)
        
        # Open the image using PIL and convert to RGB
        img = Image.open(image_buffer).convert('RGB')
        
        return img

    def predict(self, request: Dict) -> Generator[str, None, None]:
        """
        Generate text based on image and prompt with streaming output.
        
        Args:
            request: Dictionary with keys:
                - 'system': System prompt for guidance
                - 'prompt': Text query about the image
                - 'image_base64' or 'image': Image input (base64 string or path)
                
        Returns:
            Generator yielding text tokens as they're generated
        """
        try:
            # Get the system prompt and user prompt
            system = request.get("system", "")
            prompt = request.get("prompt", "Describe this image.")
            
            # Process the image source
            if "image_base64" in request:
                # Convert base64 to PIL Image
                image = self.decode_base64_to_image(request["image_base64"])
            elif "image" in request:
                image_source = request["image"]
                if isinstance(image_source, str):
                    if image_source.startswith("data:image") or "base64" in image_source:
                        # Handle data URL
                        base64_data = image_source.split(",")[1] if "," in image_source else image_source
                        image = self.decode_base64_to_image(base64_data)
                    elif os.path.exists(image_source):
                        # Handle file path
                        image = Image.open(image_source).convert('RGB')
                    else:
                        raise ValueError(f"Invalid image source: {image_source}")
                elif isinstance(image_source, Image.Image):
                    # Handle PIL Image directly
                    image = image_source
                else:
                    raise ValueError(f"Unsupported image source type: {type(image_source)}")
            else:
                raise ValueError("No image provided in request")
            
            # Build the conversation message structure
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Prepare the text prompt
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process images without videos
            image_inputs = [image]
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Create a streamer for the generated text
            streamer = TextIteratorStreamer(
                self.processor.tokenizer, skip_prompt=True, skip_special_tokens=False
            )
            
            # Set up generation configuration
            generation_config = GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,
            )
            
            # Prepare generation keyword arguments
            generation_kwargs = {
                **inputs,
                "generation_config": generation_config,
                "max_new_tokens": self.max_tokens,
                "streamer": streamer,
            }
            
            # Start the generation in a separate thread
            with torch.no_grad():
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()
                
                def inner():
                    try:
                        for token in streamer:
                            if token.strip():  # Ensure token is not empty
                                yield token
                    except StopIteration:
                        pass
                    except Exception as e:
                        print(f"Error in token streaming: {e}")
                    finally:
                        thread.join()  # Ensure thread is properly joined
                
                return inner()
                
        except Exception as e:
            print(f"Error in predict method: {e}")
            def error_generator():
                yield f"Error processing request: {str(e)}"
            return error_generator()


# Example usage
if __name__ == "__main__":
    # Example configuration
    vlm_config = {
        "llm_checkpoint": "Qwen/Qwen2.5-VL-7B-Instruct",
        "llm_processor": "Qwen/Qwen2-VL-7B-Instruct",
        "llm_device": "cuda:0",
        "llm_max_tokens": 1000,
        "llm_temperature": 0.2,
        "llm_top_p": 0.95,
        "llm_top_k": 40,
    }
    
    # Initialize and load the model
    model = StreamVLM(vlm_config)
    model.load()
    
    # Example image path
    image_path = "/nvme-data/sambal/LLMVoX/assets/image_sample.png"
    
    # Load and encode image if it exists
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
        # Prepare the request
        request = {
            "image_base64": image_base64,
            "prompt": "What is this image about?",
            "system": "Answer the question in short responses."
        }
        
        # Generate and print tokens
        for token in model.predict(request):
            print(token, end="", flush=True)