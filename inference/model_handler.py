"""
ModelHandler: Manages the text-to-speech model components.

This module initializes and manages the different components needed for
the streaming TTS system, including the WavTokenizer, text encoder, and 
speech generation model.
"""
import os
import torch
from typing import Dict, Optional, Tuple
from transformers import T5ForConditionalGeneration, AutoTokenizer
from WavTokenizer.decoder.pretrained import WavTokenizer
from src.model import GPTConfig, GPT
from .llm_streaming import StreamModel
from .multimodal_streaming import StreamMultimodalModel
try:
    from .vlm_streaming import StreamVLM
except:
    print("Update transformers version for Qwen 2.5 VL")
    pass

def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict, tokenizer, model):
    """
    Resize tokenizer and embedding with smart initialization.
    
    Args:
        special_tokens_dict: Dictionary with special tokens to add
        tokenizer: Tokenizer to modify
        model: Model with embeddings to resize
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class ModelHandler:
    """Handler for loading and managing TTS model components."""
    
    def __init__(self,config,device_id: Optional[int] = None):
        """
        Initialize the model handler on a specific device.
        
        Args:
            device_id: GPU device ID to use, or None for CPU
        """
        self.config=config
        device_str = f"cuda:{device_id}" if torch.cuda.is_available() and device_id is not None else "cpu"
        self.device = torch.device(device_str)
        print(f"Initializing model handler on {self.device}")
        
        # Load each component
        self.wavtokenizer = self.initialize_wavtokenizer()
        self.tokenizer, self.llm_model = self.initialize_llm_model()
        self.model = self.initialize_gpt_model()
        

    def initialize_wavtokenizer(self) -> WavTokenizer:
        """
        Initialize and load the WavTokenizer model.
        
        Returns:
            Initialized WavTokenizer
        """
        print(f"Loading WavTokenizer from {self.config['wav_model_path']} on {self.device}")
        wavtokenizer = WavTokenizer.from_pretrained0802(
            self.config['wav_config_path'], 
            self.config['wav_model_path']
        )
        return wavtokenizer.to(self.device)

    def initialize_llm_model(self) -> Tuple[AutoTokenizer, torch.nn.Module]:
        """
        Initialize the encoder model and tokenizer.
        
        Returns:
            Tuple of (tokenizer, embedding model)
        """
        print(f"Loading encoder model from {self.config['encoder_model_path']}")
        llm_model = T5ForConditionalGeneration.from_pretrained(self.config['encoder_model_path'])
        tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer_path'])
        
        # Add special tokens
        special_tokens = [
            dict(pad_token="[PAD]"),
            dict(pad_token="EOS"),
        ]
        
        for tokens in special_tokens:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=tokens,
                tokenizer=tokenizer,
                model=llm_model,
            )
        
        # Use only the encoder's embedding layer
        llm_model = llm_model.encoder.embed_tokens.to(self.device)
        return tokenizer, llm_model

    def initialize_stream_model(self) -> StreamModel:
        """
        Initialize the streaming LLM model.
        
        Returns:
            Initialized StreamModel
        """
        print("Initializing streaming LLM model")
        stream_model = StreamModel(self.config)
        stream_model.load()
        return stream_model

    def initialize_vlm_model(self) -> StreamModel:
        """
        Initialize the streaming LLM model.
        
        Returns:
            Initialized StreamModel
        """
        print("Initializing streaming LLM model")
        stream_model = StreamVLM(self.config)
        stream_model.load()
        return stream_model


    def initialize_stream_multimodal(self) -> StreamMultimodalModel:

        stream_model = StreamMultimodalModel(self.config)
        stream_model.load()
        return stream_model


    def initialize_gpt_model(self) -> GPT:
        """
        Initialize the GPT model for speech generation.
        
        Returns:
            Initialized GPT model
        """
        print(f"Loading GPT model from {self.config['llmvox_checkpoint_path']}")
        checkpoint = torch.load(self.config['llmvox_checkpoint_path'], map_location=self.device)
        checkpoint_model_args = checkpoint['model_args']
        model_args = {k: checkpoint_model_args[k] for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']}
        model_args["is_train"]=False
        # Create model with loaded config
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
        # Load state dict from checkpoint
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model