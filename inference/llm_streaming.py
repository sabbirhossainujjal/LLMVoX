"""
LLM-Streaming
StreamModel: A class for text generation with streaming output.

This module provides a streaming interface to LLMs like Llama, allowing
token-by-token generation for real-time applications.
"""
from threading import Thread

from typing import Dict, Generator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextIteratorStreamer

class StreamModel:
    """Streaming text generation model wrapper."""
    
    def __init__(self,config) -> None:
        """Initialize model with empty tokenizer and model attributes."""
        self.config=config
        self.tokenizer = None
        self.model = None
        self.device =config["llm_device"]

    def load(self):
        """
        Load the tokenizer and model.
        
        The model is loaded onto the specified device.
        """
        print(f"Loading LLM from {self.config['llm_checkpoint']}...")
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["llm_checkpoint"])
        
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["llm_checkpoint"],
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",  # Enable Flash Attention
            use_cache=True,              # Enable KV cache
        ).to(self.device)
        
        print(f"LLM loaded successfully on {self.device}")

    def predict(self, request: Dict) -> Generator[str, None, None]:
        """
        Generate text using streaming output.
        
        Args:
            request: Dictionary with 'system' and 'prompt' keys
            
        Returns:
            Generator yielding text tokens as they're generated
        """
        system = request.pop("system")
        prompt = request.pop("prompt")
        prompt = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        
        # Tokenize the input
        input_ids = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        
        # Create a streamer for the generated text
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=False
        )
        
        # Generation configuration
        generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
        )
        
        with torch.no_grad():
            generation_kwargs = {
                "input_ids": input_ids["input_ids"],
                "generation_config": generation_config,
                "return_dict_in_generate": True,
                "output_scores": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "max_new_tokens": self.config["llm_max_tokens"],
                "streamer": streamer,
            }
            
            # Start the generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            def inner():
                try:
                    for text in streamer:
                        if text.strip():  # Ensure text is not empty
                            yield text
                except StopIteration:
                    pass
                finally:
                    thread.join()  # Ensure thread is properly joined

            return inner()