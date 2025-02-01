import numpy as np
import soundfile as sf
import torch
import requests
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer, AutoModelForTextToWaveform
import textwrap

# Constants for Groq Llama API
GROQ_API_KEY = "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"  # Replace with your Groq API Key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Replace with the correct endpoint

# 1️⃣ ASR: Convert Speech to Text
def speech_to_text(audio_path, model_name="jonatasgrosman/wav2vec2-large-xlsr-53-english"):
    """
    Convert audio to text using the Wav2Vec2 model.

    Parameters:
        audio_path (str): Path to the audio file (WAV format).
        model_name (str): Name of the ASR model.

    Returns:
        str: Transcribed text.
    """
    print("Loading ASR model...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    print(f"Transcribing audio: {audio_path}...")
    # Load audio
    audio_input, sample_rate = sf.read(audio_path)
    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    # Perform inference
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    # Decode the transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    print(f"Transcription: {transcription}")
    return transcription


# 2️⃣ LLM: Query Groq Llama API
def query_groq_llama(prompt):
    """
    Send a query to the Groq Llama3-8b-8192 API.

    Parameters:
        prompt (str): User input or query.

    Returns:
        str: AI-generated response.
    """
    system_prompt = (
        "You are an AI assistant named Dial Sathi that provides accurate and detailed answers to "
        "questions about government schemes, laws, and policies. Always respond "
        "in English unless otherwise requested. Give only text and don't include any symbols. "
        "Ensure the responses resemble a conversation, always providing brief answers (around 50 words) with a conclusion."
    )

    combined_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

    print(f"Querying Groq Llama API with prompt:\n{combined_prompt}")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
        "temperature": 0.7,
        "top_p": 0.9
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return "Sorry, I couldn't generate a response at the moment."

# 3️⃣ TTS: Convert Text to Speech (Modified for MMS-EN TTS)
def text_to_speech(text, speaker_id='EN-US', output_filename="output_eng.wav", max_chunk_length=100):
    """
    Convert text to speech using the facebook/mms-tts-eng model with chunk processing for long texts.

    Parameters:
        text (str): The text to be converted to speech.
        speaker_id (str): The speaker ID for the desired accent ('EN-US', 'EN-BR', etc.).
        speed (float): The speed of the speech (1.0 is normal speed).
        output_filename (str): Path to save the generated audio file.
        max_chunk_length (int): Maximum length of text chunks for processing.

    Returns:
        str: Path to the generated audio file.
    """
    print("Loading MMS-EN TTS model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    model = AutoModelForTextToWaveform.from_pretrained("facebook/mms-tts-eng")

    print(f"Splitting text into chunks of max length {max_chunk_length}...")
    chunks = textwrap.wrap(text, max_chunk_length)

    audio_segments = []

    # Process each chunk
    for chunk in chunks:
        print(f"Processing chunk: {chunk[:20]}...")
        inputs = tokenizer(chunk, return_tensors="pt")

        # Generate the waveform for the chunk
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], speaker_id=speaker_id)

        # Extract the audio waveform (model will return a tensor)
        audio_arr = outputs.waveform.squeeze().cpu().numpy()  # assuming 'waveforms' is the output key
        audio_segments.append(audio_arr)

    # Concatenate all audio segments into a single waveform
    print("Concatenating audio segments...")
    final_audio = np.concatenate(audio_segments)

    # Save the audio file
    sf.write(output_filename, final_audio, 16000)  # Assuming 16kHz sample rate for the MMS-EN model
    print(f"Generated audio saved at: {output_filename}")

    return output_filename

# 4️⃣ Integration: ASR → LLM → TTS Pipeline
def speech_to_speech_pipeline(input_audio, speaker_id="EN-US"):
    """
    End-to-end pipeline to process input speech and generate a response in speech.

    Parameters:
        input_audio (str): Path to the input audio file (WAV format).
        speaker_id (str): Speaker ID for TTS output.
        speed (float): Speech playback speed.

    Returns:
        str: Path to the generated response audio file.
    """
    print("Starting speech-to-speech pipeline...")

    # Step 1: ASR (Speech to Text)
    text = speech_to_text(input_audio)

    # Step 2: LLM (Generate Response)
    response = query_groq_llama(text)

    # Step 3: TTS (Text to Speech with MMS-EN TTS)
    output_audio = text_to_speech(response, output_filename="response.wav")
    return output_audio
def main():
    # Input audio file recorded by the user
    input_audio_file = input(
        "Enter the path to your audio file (e.g., 'audio.wav'): ").strip()  # Replace with your input audio path
    output_audio_file = speech_to_speech_pipeline(input_audio_file, speaker_id="EN-US")

    print(f"Response audio saved at: {output_audio_file}")
# Example Usage
if __name__ == "__main__":
   main()
