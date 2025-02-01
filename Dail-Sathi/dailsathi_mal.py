import os
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
from scipy.io import wavfile
from pydub import AudioSegment
import requests
import numpy as np
import textwrap
import soundfile as sf
from IndicTransToolkit import IndicProcessor


# Device setup (CUDA if available, otherwise CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use the Amrrs Wav2Vec2 ASR model
ASR_PIPELINE = pipeline(
    task="automatic-speech-recognition",
    model="cdactvm/w2v-bert-malayalam-v2",
    device=0 if torch.cuda.is_available() else -1
)
# Groq API URL
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = "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"  # Replace with your Groq API key

def initialize_translation_model():
    """Set up the translation model and tokenizer."""
    src_lang, tgt_lang = "mal_Mlym", "eng_Latn"
    model_name = "ai4bharat/indictrans2-indic-en-1B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    ).to(DEVICE)

    ip = IndicProcessor(inference=True)

    return model, tokenizer, ip, src_lang, tgt_lang


TRANSLATION_MODEL, TRANSLATION_TOKENIZER, INDIC_PROCESSOR, SRC_LANG, TGT_LANG = initialize_translation_model()


def translate_text_with_model(text, model, tokenizer, ip, source_lang, target_lang):
    """Translate text using the initialized model, tokenizer, and processor."""
    batch = ip.preprocess_batch([text], src_lang=source_lang, tgt_lang=target_lang)

    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    with tokenizer.as_target_tokenizer():
        generated_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    translations = ip.postprocess_batch(generated_tokens, lang=target_lang)
    return translations[0]

def initialize_translation_model_en():
    """Set up the translation model and tokenizer."""
    src_lang, tgt_lang = "eng_Latn", "mal_Mlym"
    model_name = "ai4bharat/indictrans2-en-indic-1B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    ).to(DEVICE)

    ip = IndicProcessor(inference=True)

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

def translate_text_with_model_en(text, model, tokenizer, ip, source_lang, target_lang):
    """Translate text using the initialized model, tokenizer, and processor."""
    batch = ip.preprocess_batch([text], src_lang=source_lang, tgt_lang=target_lang)

    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    with tokenizer.as_target_tokenizer():
        generated_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    translations = ip.postprocess_batch(generated_tokens, lang=target_lang)
    return translations[0]

# Text-to-Speech Function
def text_to_speech(text, output_filename :str, speaker_id = 11, style_id = 0):
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
            print(f"Original Transcription in Malayalam: {transcription}")

            # Translate transcription from Telugu to English
            translated_text = translate_text_with_model(
                transcription,
                TRANSLATION_MODEL,
                TRANSLATION_TOKENIZER,
                INDIC_PROCESSOR,
                SRC_LANG,
                TGT_LANG
            )
            print(f"Translated Text (Malayalam to English): {translated_text}")
        else:
            print("No transcription available.")
            translated_text = None

        if translated_text:
            # Query the LLM with the English translation
            llm_response = query_groq_llama(translated_text)
            print(f"LLM Response in English: {llm_response}")

            # Translate the LLM response back to Kannada
            translated_text_mal = translate_text_with_model_en(
                llm_response,
                TRANSLATION_MODEL1,
                TRANSLATION_TOKENIZER1,
                INDIC_PROCESSOR1,
                SRC_LANG1,
                TGT_LANG1
            )
            print(f"LLM Response Translated Back to Malayalam: {translated_text_mal}")

            # Pass the Malayalam response to TTS
            output_audio_file = text_to_speech(translated_text_mal, output_filename=f"{unique_filename}_output.wav")

            return str(output_audio_file)
        else:
            print("No translated text available.")

# if __name__ == "__main__":
#     main()