import os
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer,AutoModelForTextToWaveform
from scipy.io import wavfile
from pydub import AudioSegment
import requests
import numpy as np
from IndicTransToolkit import IndicProcessor
import textwrap
import soundfile as sf

# Device setup (CUDA if available, otherwise CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration for ASR and Groq LLM
ASR_PIPELINE = pipeline(
    "automatic-speech-recognition",
    model="parthiv11/indic_whisper_hi_multi_gpu"
)

TTS_TOKENIZER = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
TTS_MODEL = AutoModelForTextToWaveform.from_pretrained("facebook/mms-tts-hin").to(DEVICE)

# Groq API URL
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = "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"  # Replace with your Groq API key

# Initialize translation model components
def initialize_translation_model():
    """Set up the translation model and tokenizer."""
    src_lang, tgt_lang = "hin_Deva", "eng_Latn"
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
def initialize_translation_model_en():
    """Set up the translation model and tokenizer."""
    src_lang, tgt_lang = "eng_Latn", "hin_Deva"
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


# Audio Processing Functions
def reprocess_audio(input_filename, output_filename="processed.wav"):
    """Reprocess the WAV file to ensure correct formatting."""
    print(f"Reprocessing audio file: {input_filename}")
    audio = AudioSegment.from_file(input_filename)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(output_filename, format="wav")
    print(f"Audio repr  ocessed and saved as {output_filename}")
    return output_filename


def transcribe_audio(input_filename):
    """Transcribe audio using the Hugging Face ASR pipeline."""
    print("Transcribing audio using Hugging Face pipeline...")
    sample_rate, audio_data = wavfile.read(input_filename)

    if audio_data.dtype == np.int16:
        audio_data = audio_data / 32768.0

    result = ASR_PIPELINE({"array": audio_data, "sampling_rate": sample_rate})
    transcription = result.get("text", "")
    print(f"Transcription: {transcription}")
    return transcription


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

'''
# Initialize the TTS model and tokenizers
def initialize_tts_model():
    """Initialize the TTS model and tokenizers."""
    model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    return model, tokenizer, description_tokenizer


TTS_MODEL, TTS_TOKENIZER, DESCRIPTION_TOKENIZER = initialize_tts_model()
'''


def hindi_text_to_speech(hindi_text, output_filename="indic_tts_out.wav"):
    """
    Convert a long Hindi text into speech using the MMS-TTS model.

    Args:
        hindi_text (str): Translated LLM response in Hindi.
        output_filename (str): Path to save the generated audio.

    Returns:
        str: Path to the generated audio file.
    """
    # Split the Hindi text into manageable chunks
    max_chunk_length = 50  # Adjust this based on the model's input capacity
    chunks = textwrap.wrap(hindi_text, max_chunk_length)

    # Generate audio for each chunk
    audio_segments = []
    for chunk in chunks:
        print(f"Processing chunk: {chunk[:20]}...")
        inputs = TTS_TOKENIZER(chunk, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            audio_arr = TTS_MODEL(**inputs).waveform

        audio_arr = audio_arr.cpu().numpy().squeeze()
        audio_segments.append(audio_arr)

    final_audio = np.concatenate(audio_segments)

    # Save the audio file
    sf.write(output_filename, final_audio, 16000)  # MMS-TTS uses a 16kHz sample rate
    print(f"Generated audio saved at: {output_filename}")

    return output_filename

def main():
    raw_audio_file = input("Enter the path to your audio file (e.g., 'audio.wav'): ").strip()

    if not os.path.exists(raw_audio_file):
        print("Error: File does not exist. Please check the file path.")
    else:
        processed_audio_file = reprocess_audio(raw_audio_file)
        transcription = transcribe_audio(processed_audio_file)

        if transcription:
            translated_text = translate_text_with_model(
                transcription,
                TRANSLATION_MODEL,
                TRANSLATION_TOKENIZER,
                INDIC_PROCESSOR,
                SRC_LANG,
                TGT_LANG
            )
        else:
            print("No transcription available.")
            translated_text = None

        if translated_text:
            llm_response = query_groq_llama(translated_text)
            print(f"Final Response: {llm_response}")
            translated_text_hi = translate_text_with_model_en(
                llm_response,
                TRANSLATION_MODEL1,
                TRANSLATION_TOKENIZER1,
                INDIC_PROCESSOR1,
                SRC_LANG1,
                TGT_LANG1
            )
            print("Now Printing the llm response in hindi")
            print(f"LLM response: {translated_text_hi}")
            hindi_text_to_speech(translated_text_hi)

# Main Function
if __name__ == "__main__":
    main()