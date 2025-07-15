import whisper
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

access_token = os.getenv("ACCESS_TOKEN")

def load_transcription_models():
    # Whisper model
    # whisper_model = whisper.load_model("large-v3")
    whisper_model = whisper.load_model("small")
    print("Whisper model loaded.")

    # Silero VAD model
    print("Loading Silero VAD model...")
    vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
    vad_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vad_model.to(vad_device)
    (get_speech_timestamps, _, _, _, _) = vad_utils
    print("Silero VAD model loaded.")
    return whisper_model, vad_model, get_speech_timestamps, vad_device

def load_summarization_model():
    # Load Llama model for summarization
    print("Loading summariser model...")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", return_full_text=False)
    print("summariser model loaded.")
    return summarizer
