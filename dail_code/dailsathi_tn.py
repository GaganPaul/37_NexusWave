import os
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
from scipy.io import wavfile
from pydub import AudioSegment
import requests
import numpy as np
#from IndicTransToolkit import IndicProcessor
import textwrap
import soundfile as sf
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


# Device setup (CUDA if available, otherwise CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Use the Amrrs Wav2Vec2 ASR model
ASR_PIPELINE = pipeline(
    task="automatic-speech-recognition",
    model="Amrrs/wav2vec2-large-xlsr-53-tamil",
    device=0 if torch.cuda.is_available() else -1
)
# Groq API URL
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = "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"  # Replace with your Groq API key

def initialize_translation_model():
    """Set up the translation model and tokenizer."""
    model_name = "facebook/mbart-large-50-many-to-many-mmt"

    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(
        model_name,
    ).to(DEVICE)

    return model, tokenizer


TRANSLATION_MODEL, TRANSLATION_TOKENIZER = initialize_translation_model()



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

    # Decode the generated tokens to text
    translations = tokenizer.batch_decode(
        generated_tokens.detach().cpu(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return translations[0]

def initialize_translation_model_en():
    """Set up the translation model and tokenizer."""
    src_lang, tgt_lang = "eng_Latn", "ta_Taml"
    model_name = "Mr-Vicky-01/English-Tamil-Translator"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(DEVICE)

    ip = None  # Placeholder since the new model does not use IndicProcessor.

    return model, tokenizer, ip, src_lang, tgt_lang

TRANSLATION_MODEL1, TRANSLATION_TOKENIZER1, INDIC_PROCESSOR1, SRC_LANG1, TGT_LANG1 = initialize_translation_model_en()

def reprocess_audio(input_filename, output_filename="processed.wav"):
    """Reprocess the WAV file to ensure correct formatting."""
    output_dir = "./audio/audio_preprocessing/"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    output_path = os.path.join(output_dir, output_filename)

    print(f"Reprocessing audio file: {input_filename}")
    audio = AudioSegment.from_file(input_filename)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(output_path, format="wav")

    print(f"Audio reprocessed and saved as {output_path}")
    return output_path

def transcribe_audio(input_filename):
    """Transcribe audio using the Hugging Face ASR pipeline."""
    print("Transcribing audio using Hugging Face pipeline...")
    sample_rate, audio_data = wavfile.read(input_filename)

    print(f"Sample rate: {sample_rate}, Audio shape: {audio_data.shape}, Audio dtype: {audio_data.dtype}")

    # Normalize audio if it is in int16 format
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

# Groq Llama3-8b-8192 Query Function
def query_groq_llama(prompt):
    """Send a query to the Groq Llama3-8b-8192 API."""
    system_prompt = (
        "You are an AI assistant named Dial Sathi that provides accurate and detailed answers to "
        "questions about government schemes, laws, and policies. Always respond "
        "in English unless otherwise requested. give only text and don't ever give any symbols or anything and make sure the responses are like a conversion  and always give brief answers and make sure the text has a conclusion and always respond in 50 words"
    )

    combined_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

    print(f"Querying Llama3-8b-8192 API with prompt:\n{combined_prompt}")
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
        return None

def translate_text_with_model_en(text, model, tokenizer):
    """
    Translate text from English to Tamil using the initialized model and tokenizer.

    Args:
        text (str): Text to translate.
        model: Translation model (e.g., English to Tamil).
        tokenizer: Translation tokenizer.

    Returns:
        str: Translated text.
    """
    # Preprocess text using the tokenizer
    inputs = tokenizer(
        [text],
        truncation=True,
        padding="longest",
        return_tensors="pt",
    ).to(DEVICE)

    # Generate translation using the model
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    # Decode the generated tokens to get the translated text
    translation = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return translation

# Text-to-Speech Function
def text_to_speech(text, output_filename :str,speaker_id = 18, style_id = 0):
    """
    Convert text to speech using the ai4bharat/vits_rasa_13 model.

    Parameters:
        text (str): The text to be converted to speech.
        speaker_id (int): The speaker ID for the desired language and gender.
        style_id (int): The style ID for the desired speech style.
        output_filename (str): The name of the output WAV file.

    Returns:
        str: Path to the generated audio file.
    """
    # Load the TTS model and tokenizer
    print("Loading TTS model and tokenizer...")
    model = AutoModel.from_pretrained("ai4bharat/vits_rasa_13", trust_remote_code=True).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/vits_rasa_13", trust_remote_code=True)

    # Split the input text into manageable chunks
    max_chunk_length = 50  # Adjust based on the model's input capacity
    chunks = textwrap.wrap(text, max_chunk_length)

    audio_segments = []

    # Process each chunk
    for chunk in chunks:
        print(f"Processing chunk: {chunk[:20]}...")
        inputs = tokenizer(text=chunk, return_tensors="pt").to("cuda")

        # Generate the waveform for the chunk
        with torch.no_grad():
            outputs = model(inputs['input_ids'], speaker_id=speaker_id, emotion_id=style_id)

        # Extract the audio waveform
        audio_arr = outputs.waveform.cpu().numpy().squeeze()
        audio_segments.append(audio_arr)

    # Concatenate all audio segments into a single waveform
    final_audio = np.concatenate(audio_segments)

    # Ensure the output directory exists
    output_dir = "audio/output_audio"
    os.makedirs(output_dir, exist_ok=True)

    # Combine the folder and filename to get the full output path
    full_output_path = os.path.join(output_dir, output_filename)

    # Save the audio file
    sf.write(full_output_path, final_audio, model.config.sampling_rate)
    print(f"Generated audio saved at: {full_output_path}")

    return full_output_path


def main(path :str,unique_filename :str):
    # Main Function
    raw_audio_file = path

    if not os.path.exists(raw_audio_file):
        print("Error: File does not exist. Please check the file path.")
    else:
        # Reprocess the audio file
        processed_audio_file = reprocess_audio(raw_audio_file)

        # Transcribe the audio
        transcription = transcribe_audio(processed_audio_file)

        if transcription:
            print(f"Original Transcription in Tamil: {transcription}")

            # Translate transcription from Tamil to English
            translated_text = translate_text_with_model(
                transcription,
                TRANSLATION_MODEL,
                TRANSLATION_TOKENIZER
            )
            print(f"Translated Text (Tamil to English): {translated_text}")
        else:
            print("No transcription available.")
            translated_text = None

        if translated_text:
            # Query the LLM with the English translation
            llm_response = query_groq_llama(translated_text)
            print(f"LLM Response in English: {llm_response}")

            # Translate the LLM response back to Tamil
            translated_text_ta = translate_text_with_model_en(
                llm_response,
                TRANSLATION_MODEL1,
                TRANSLATION_TOKENIZER1
            )
            print(f"LLM Response Translated Back to Tamil: {translated_text_ta}")

            # Pass the Tamil response to TTS
            output_audio_file = text_to_speech(translated_text_ta, output_filename=f"{unique_filename}_output.wav")

            return str(output_audio_file)
        else:
            print("No translated text available.")

# if __name__ == "__main__":
#     main()