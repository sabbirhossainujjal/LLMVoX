import requests
import time
import threading
import queue
import base64
from threading import Thread
from pyaudio import PyAudio, paInt16, paFloat32

def stream_audio_from_api(endpoint, payload, sample_rate=24000, channels=1, chunk_size=10000):
    """
    Generic function to stream audio from any API endpoint.
    
    Args:
        endpoint (str): API endpoint URL
        payload (dict): JSON payload to send to the API
        sample_rate (int): Audio sample rate in Hz
        channels (int): Number of audio channels
        chunk_size (int): Size of audio chunks to process
        
    Returns:
        None
    """
    audio_queue = queue.Queue()  # Buffer for audio playback
    audio_chunks = []  # Buffer to store audio for saving
    start = time.time()
    
    def stream_audio():
        """Streams audio from the server and enqueues it for playback and saving."""
        try:
            with requests.post(endpoint, json=payload, stream=True) as stream:
                stream.raise_for_status()  # Raise an error for bad status codes
                for chunk in stream.iter_content(chunk_size=None):
                    if chunk:
                        try:
                            # Enqueue the chunk for playback
                            print(f"Got chunk at {time.time()-start}")
                            audio_queue.put(chunk, timeout=1)
                            # Store the chunk for saving
                            audio_chunks.append(chunk)
                        except queue.Full:
                            print("Audio queue is full. Dropping chunk.")
        except requests.exceptions.RequestException as e:
            print(f"RequestException: {e}")
        finally:
            # Signal the end of streaming
            audio_queue.put(None)

    def play_audio():
        """Plays audio chunks from the queue using PyAudio."""
        p = PyAudio()
        try:
            player = p.open(format=paFloat32,
                          channels=channels,
                          rate=sample_rate,
                          output=True,)

            while True:
                chunk = audio_queue.get()
                if chunk is None:
                    print("End of streaming.")
                    break  # End of streaming
                if not chunk:
                    print("Received an empty chunk. Skipping.")
                    continue  # Skip empty chunks

                try:
                    print("Playing chunk")
                    player.write(chunk)
                except Exception as e:
                    print(f"Error during playback: {e}")
                    break
        finally:
            player.stop_stream()
            player.close()
            p.terminate()

    # Start streaming and playback in separate threads
    stream_thread = threading.Thread(target=stream_audio, daemon=True)
    play_thread = threading.Thread(target=play_audio, daemon=True)

    stream_thread.start()
    play_thread.start()

    # Wait for both threads to finish
    stream_thread.join()
    play_thread.join()

def load_base64_from_image(image_path):
    """
    Load an image from a file and convert it to base64.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64-encoded image string
    """
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    return image_base64

def tts_stream(text, server_ip="10.127.30.115", server_port="5003"):
    """
    Convert text to speech and stream the audio.
    
    Args:
        text (str): Text to convert to speech
        server_ip (str): IP address of the server
        server_port (str): Port number for the server
    """
    endpoint = f'http://{server_ip}:{server_port}/tts'
    payload = {"text": text}
    stream_audio_from_api(endpoint, payload)

def asr(audio_data_base64, server_ip="10.127.30.115", server_port="5002", src='English', tgt='English'):
    """
    Perform speech recognition on audio data.
    
    Args:
        audio_data_base64 (str): Base64-encoded audio data
        server_ip (str): IP address of the server
        server_port (str): Port number for the server
        src (str): Source language
        tgt (str): Target language
        
    Returns:
        str: Recognized text
    """
    endpoint = f"http://{server_ip}:{server_port}/stt2"
    payload = {
        'source_language': src,
        'target_language': tgt,
        'audio_base64': audio_data_base64
    }
    response = requests.post(endpoint, json=payload).text
    return response.strip()

def voicechat(audio_data_base64, server_ip="10.127.30.115", server_port="5003", src='English', tgt='English'):
    """
    Process audio input and generate a voice response.
    
    Args:
        audio_data_base64 (str): Base64-encoded audio data
        server_ip (str): IP address of the server
        server_port (str): Port number for the server
        src (str): Source language
        tgt (str): Target language
    """
    endpoint = f'http://{server_ip}:{server_port}/voicechat'
    payload = {
        'source_language': src,
        'target_language': tgt,
        'audio_base64': audio_data_base64
    }
    stream_audio_from_api(endpoint, payload)

def vlmschat(audio_data_base64, image_path, server_ip="10.127.30.115", server_port="5003", src='English', tgt='English'):
    """
    Process audio input with visual context and generate a voice response.
    
    Args:
        audio_data_base64 (str): Base64-encoded audio data
        image_path (str): Path to the image file
        server_ip (str): IP address of the server
        server_port (str): Port number for the server
        src (str): Source language
        tgt (str): Target language
    """
    endpoint = f'http://{server_ip}:{server_port}/vlmschat'
    image_raw = load_base64_from_image(image_path)
    payload = {
        'source_language': src,
        'target_language': tgt,
        'audio_base64': audio_data_base64,
        'image_base64': image_raw
    }
    stream_audio_from_api(endpoint, payload)

def multimodalchat(audio_data_base64, image_path, server_ip="10.127.30.115", server_port="5003", src='English', tgt='English'):
    """
    Process audio input with multiple images and generate a voice response.
    
    Args:
        audio_data_base64 (str): Base64-encoded audio data
        image_path (str): Path to the image file
        server_ip (str): IP address of the server
        server_port (str): Port number for the server
        src (str): Source language
        tgt (str): Target language
    """
    endpoint = f'http://{server_ip}:{server_port}/multimodalchat'
    image_raw = load_base64_from_image(image_path)
    payload = {
        'audio_base64': audio_data_base64,
        'image_list': [image_raw],
        'source_language': src,
        'target_language': tgt
    }
    stream_audio_from_api(endpoint, payload)