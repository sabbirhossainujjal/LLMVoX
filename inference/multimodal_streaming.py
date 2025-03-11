"""
StreamMultimodalModel: A class for multimodal (text, audio, image) generation with streaming output.

This module provides a streaming interface to Phi-4-multimodal-instruct model,
allowing token-by-token generation for real-time applications with audio and image inputs.
"""
from threading import Thread
from typing import Dict, Generator, List, Tuple, Optional, Union
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig, TextIteratorStreamer
from PIL import Image
import soundfile as sf
import base64
import numpy as np
import base64
from io import BytesIO
from PIL import Image

class StreamMultimodalModel:
    """Streaming text generation model wrapper for Phi-4-multimodal-instruct."""
    
    def __init__(self,config) -> None:
        """Initialize model with empty processor and model attributes."""
        
        self.config=config
        self.processor = None
        self.model = None
        self.device = config['llm_device']
        self.model_path = config["llm_checkpoint"]
        
    def load(self):
        """
        Load the processor and model.
        
        The model is loaded onto the specified device.
        """
        print(f"Loading Phi-4 Multimodal LLM from {self.model_path}...")
        
        # Load the processor (handles tokenization and multimodal processing)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            trust_remote_code=True,
        )
        
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_cache=True  # Enable KV cache
        ).to(self.device)
        
        # Load generation config
        self.generation_config = GenerationConfig.from_pretrained(self.model_path)
        
        print(f"Phi-4 Multimodal LLM loaded successfully on {self.device}")
    
    def predict(self, request: Dict) -> Generator[str, None, None]:
        """
        Generate text using streaming output with multimodal inputs.
        
        Args:
            request: Dictionary with keys:
                'system': system message
                'prompt': optional text prompt
                'audio_files': optional list of paths to audio files (.wav)
                'images': optional list of paths to image files
                
        Returns:
            Generator yielding text tokens as they're generated
        """
        system = request.get("system", "You are a helpful assistant.")
        user_text = request.get("prompt", "")
        
        # Format the chat template strictly per Phi-4 documentation
        system_prompt = f"<|system|>{system}<|end|>"
        user_prompt = "<|user|>"
        
        # Process images if provided
        images = []


        if "images_data" in request and request["images_data"]:
            for image_string in request["images_data"]:
                img = decode_base64_to_image(image_string)
                images.append(img)
                user_prompt += f"<|image_{len(images)}|>"

        if "images_files" in request and request["images_files"]:
            for image_path in request["images_files"]:
                img = Image.open(image_path).convert('RGB')
                images.append(img)
                user_prompt += f"<|image_{len(images)}|>"
        
        # Process audio files if provided
        audio_data = []
        if "audio_files" in request and request["audio_files"]:
            for audio_path in request["audio_files"]:
                if not audio_path.lower().endswith('.wav'):
                    print(f"Warning: {audio_path} is not a .wav file. Only WAV files are fully supported.")
                audio, samplerate = sf.read(audio_path)
                audio_data.append((audio, samplerate))
                user_prompt += f"<|audio_{len(audio_data)}|>"

        if "audio_data" in request and request["audio_data"]:
            samplerate=16000
            audio_array = base64.b64decode(request['audio_data'])
            audio_array= np.frombuffer(audio_array, dtype=np.int16)
            audio_array = audio_array.astype(np.float32)
            audio_array /= np.max(np.abs(audio_array))
            
            audio_data.append((audio_array, samplerate))
            user_prompt += f"<|audio_1|>"
        
        if "images_files" not in request and "images_data" not in request:
            user_text = "Based on the attached audio, answer the query in a descriptive manner."
        
        # Add user text and finish the prompt
        user_prompt += f"{user_text}<|end|>"
        assistant_prompt = "<|assistant|>"
        
        # Combine all parts
        full_prompt = f"{system_prompt}{user_prompt}{assistant_prompt}"
        
        # Process inputs with the processor
        processor_inputs = {"text": full_prompt}
        if images:
            processor_inputs["images"] = images
        if audio_data:
            processor_inputs["audios"] = audio_data
        
        # print(full_prompt)
        # Tokenize and prepare inputs for the model
        inputs = self.processor(**processor_inputs, return_tensors="pt").to(self.device)
        
        # print(inputs)
        # Create a streamer for the generated text
        streamer = TextIteratorStreamer(
            self.processor, 
            skip_prompt=True, 
            skip_special_tokens=False
        )
        
        inputs['generation_config']=self.generation_config
        inputs["max_new_tokens"]=self.config["llm_max_tokens"]
        inputs["streamer"]=streamer
        inputs['do_sample']=True
        inputs['temperature']=self.config["llm_temperature"]

        # Start the generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=inputs)
        thread.start()
        
        def inner():
            try:
                for text in streamer:
                    if text.strip():  # Ensure text is not empty
                        yield text
            except Exception as e:
                print(f"Error during streaming: {e}")
                import traceback
                traceback.print_exc()
            finally:
                thread.join()  # Ensure thread is properly joined
        
        return inner()

def load_base64_from_image(image_path):
    # Load the image from the specified path
    with open(image_path, "rb") as image_file:
        # Read the image and encode it to base64
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    return image_base64


def decode_base64_to_image(base64_string):
    # Decode the base64 string back to bytes
    image_bytes = base64.b64decode(base64_string)
    
    # Create a BytesIO object from the decoded bytes
    image_buffer = BytesIO(image_bytes)
    
    # Open the image using PIL and convert to RGB
    img = Image.open(image_buffer).convert('RGB')
    
    return img

# Example usage
if __name__ == "__main__":

    import torch
    config = {
        # Model paths
        "wav_config_path": "./WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        "wav_model_path": "./CHECKPOINTS/wavtokenizer_large_speech_320_24k.ckpt",
        "encoder_model_path": "charsiu/g2p_multilingual_byT5_tiny_16_layers_100",
        "tokenizer_path": "google/byt5-small",
        "gpt_checkpoint_path": "./CHECKPOINTS/ckpt_english_tiny.pt",
        
        # LLM settings
    #"llm_checkpoint": "meta-llama/Llama-3.1-8B-Instruct",
        "llm_checkpoint": "microsoft/Phi-4-multimodal-instruct",
        "llm_device": "cuda:0",
        "llm_max_tokens": 1000,
        "llm_temperature": 0.7,
        "llm_top_p": 0.95,
        "llm_top_k": 40,
        
        # TTS model settings
        "tts_device_1": 1, #GPU ID
        "tts_device_2": 1,
        
        # Streaming settings
        "system_prompt": "You are a friendly assistant that answers questions concisely. Do not answer in list, use brackets etc. Think yourself as person speaking.",
        "initial_dump_size_1": 40,
        "initial_dump_size_2": 160,
        "max_dump_size": 1280,
        "max_audio_length": 8000,
        
        # Special tokens
        "eos_token": "<|end|>",
        # "eos_token": "<|eot_id|>",
        "pad_token_id": 384,
        "eoa_token_id": 453,
        
        # API settings
        "api_host": "0.0.0.0",
        "api_port": 5003
    }
    # Initialize the model and load it
    model = StreamMultimodalModel(config)
    model.load()

    # Define system instruction and user prompt
    system = "You are an assistant which answers queries in a conversational and a very very short responses."
    prompt = ""
    
    import time
    
    for i in range(5):
        start = time.time()

        image_raw=load_base64_from_image("/nvme-data/sambal/LLMVoX/assets/image_sample.png")

        text_streamer = model.predict({
            "system": system, 
            "prompt": "",
            "audio_files" :["/nvme-data/sambal/LLMVoX/assets/audio_sample_phi4.wav"],
            "images_data":[image_raw]
        })
        
        output = []
        count = 0
        for x in text_streamer:
            if x:
                print(x)
                output.append(x)

        print("Full response:", "".join(output))
        print(f"Total time: {time.time()-start:.2f}s")