import os
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
from scipy.io import wavfile
from pydub import AudioSegment
import requests
import numpy as np
import textwrap
import soundfile as sf
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Device setup (CUDA if available, otherwise CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Use the Amrrs Wav2Vec2 ASR model
print("Initializing ASR pipeline...")
ASR_PIPELINE = pipeline(
    task="automatic-speech-recognition",
    model="Amrrs/wav2vec2-large-xlsr-53-tamil",
    device=0 if torch.cuda.is_available() else -1
)
print("ASR pipeline initialized successfully.")

# Groq API URL
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = ""  # Replace with your Groq API key

print("Initializing translation model...")
def initialize_translation_model():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    print("Translation model loaded successfully.")
    return model, tokenizer

TRANSLATION_MODEL, TRANSLATION_TOKENIZER = initialize_translation_model()

print("Initializing English to Tamil translation model...")
def initialize_translation_model_en():
    src_lang, tgt_lang = "eng_Latn", "ta_Taml"
    model_name = "Mr-Vicky-01/English-Tamil-Translator"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16).to(DEVICE)
    print("English to Tamil translation model loaded successfully.")
    return model, tokenizer, None, src_lang, tgt_lang

TRANSLATION_MODEL1, TRANSLATION_TOKENIZER1, _, SRC_LANG1, TGT_LANG1 = initialize_translation_model_en()

def translate_text_with_model(text, model, tokenizer):
    """Translate text using the initialized model and tokenizer."""
    # Set the source language explicitly
    tokenizer.src_lang = "ta_IN"  # Tamil as the source language in MBart

    # Tokenize the input text
    inputs = tokenizer(
        text,
        truncation=True,
        padding="longest",
        max_length=2048,
        return_tensors="pt",
    ).to(DEVICE)

    # Generate translations
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],  # English as the target language
            num_beams=10,
            length_penalty=1.5,
            repetition_penalty=2.0,
            max_new_tokens=256,
            early_stopping=True,
        )

def reprocess_audio(input_filename, output_filename="processed.wav"):
    print(f"Reprocessing audio file: {input_filename}")
    output_dir = "./audio/audio_preprocessing/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    audio = AudioSegment.from_file(input_filename)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(output_path, format="wav")
    print(f"Audio reprocessed and saved as {output_path}")
    return output_path

def transcribe_audio(input_filename):
    print(f"Transcribing audio: {input_filename}")
    sample_rate, audio_data = wavfile.read(input_filename)
    print(f"Sample rate: {sample_rate}, Audio shape: {audio_data.shape}")
    if audio_data.dtype == np.int16:
        audio_data = audio_data / 32768.0
    try:
        result = ASR_PIPELINE(input_filename)
        transcription = result.get("text", "")
        print(f"Transcription: {transcription}")
        return transcription
    except Exception as e:
        print(f"ASR pipeline error: {e}")
        return None

def query_groq_llama(prompt):
    print(f"Querying Llama3-8b-8192 API with prompt: {prompt}")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {"model": "llama3-8b-8192", "messages": [{"role": "user", "content": prompt}], "max_tokens": 150, "temperature": 0.7, "top_p": 0.9}
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        llm_response = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"LLM Response: {llm_response}")
        return llm_response
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def text_to_speech(text, output_filename):
    print(f"Converting text to speech: {text}")
    model = AutoModel.from_pretrained("ai4bharat/vits_rasa_13", trust_remote_code=True).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/vits_rasa_13", trust_remote_code=True)
    chunks = textwrap.wrap(text, 50)
    audio_segments = []
    for chunk in chunks:
        print(f"Processing chunk: {chunk}")
        inputs = tokenizer(text=chunk, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(inputs['input_ids'], speaker_id=18, emotion_id=0)
        audio_arr = outputs.waveform.cpu().numpy().squeeze()
        audio_segments.append(audio_arr)
    final_audio = np.concatenate(audio_segments)
    output_dir = "audio/output_audio"
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, output_filename)
    sf.write(full_output_path, final_audio, model.config.sampling_rate)
    print(f"Generated audio saved at: {full_output_path}")
    return full_output_path

def main(path, unique_filename):
    print(f"Processing file: {path} with unique filename: {unique_filename}")
    if not os.path.exists(path):
        print("Error: File does not exist.")
        return
    processed_audio_file = reprocess_audio(path)
    transcription = transcribe_audio(processed_audio_file)
    if transcription:
        translated_text = translate_text_with_model(transcription, TRANSLATION_MODEL, TRANSLATION_TOKENIZER)
        print(f"Translated Text: {translated_text}")
        llm_response = query_groq_llama(translated_text)
        translated_text_ta = translate_text_with_model_en(llm_response, TRANSLATION_MODEL1, TRANSLATION_TOKENIZER1)
        print(f"Final Tamil Translation: {translated_text_ta}")
        output_audio_file = text_to_speech(translated_text_ta, f"{unique_filename}_output.wav")
        return str(output_audio_file)
    else:
        print("No transcription available.")
