import os
import numpy as np
import soundfile as sf
import torch
import requests
import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer, AutoModelForTextToWaveform
import textwrap

# Constants for Groq Llama API
GROQ_API_KEY = ""  # Replace with your Groq API Key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Initialize Embedding Model & FAISS Index
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dimension = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)
document_store = []  # To keep track of documents


# RAG: Document Processing & Embedding
def process_document(file_path):
    print(f"Processing document: {file_path}")
    text = ""
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    if not text:
        print("No text extracted from document.")
        return

    chunks = text.split("\n\n")
    embeddings = embedding_model.encode(chunks)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        index.add(np.array([embedding], dtype=np.float32))
        document_store.append(chunk)

    print(f"Document stored in FAISS index: {file_path}")


# RAG: Retrieve Context
def retrieve_context(query):
    query_embedding = embedding_model.encode([query]).astype(np.float32)
    if index.ntotal == 0:
        return None
    distances, indices = index.search(query_embedding, k=3)
    retrieved_docs = [document_store[i] for i in indices[0] if i < len(document_store)]
    return " ".join(retrieved_docs) if retrieved_docs else None


# ASR: Convert Speech to Text
def speech_to_text(audio_path, model_name="jonatasgrosman/wav2vec2-large-xlsr-53-english"):
    print("Loading ASR model...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    print(f"Transcribing audio: {audio_path}...")
    audio_input, sample_rate = sf.read(audio_path)
    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    print(f"Transcription: {transcription}")
    return transcription


# LLM: Query Groq Llama API with RAG
def query_groq_llama(prompt):
    context = retrieve_context(prompt)
    if context:
        print("Using retrieved RAG context...")
        prompt_with_context = f"Using the following information:\n{context}\n\nUser: {prompt}\n\nAssistant:"
    else:
        print("No relevant RAG data found. Proceeding without context.")
        prompt_with_context = prompt

    system_prompt = (
        "You are an AI assistant named Dial Sathi that provides accurate and detailed answers to "
        "questions about government schemes, laws, and policies. If context is provided, use it. "
        "Otherwise, answer based on general knowledge."
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_with_context}
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return "Sorry, I couldn't generate a response at the moment."


# TTS: Convert Text to Speech
def text_to_speech(text, speaker_id='EN-US', output_filename="output_eng.wav", max_chunk_length=100):
    print("Loading MMS-EN TTS model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    model = AutoModelForTextToWaveform.from_pretrained("facebook/mms-tts-eng")
    print(f"Splitting text into chunks of max length {max_chunk_length}...")
    chunks = textwrap.wrap(text, max_chunk_length)
    audio_segments = []

    for chunk in chunks:
        print(f"Processing chunk: {chunk[:20]}...")
        inputs = tokenizer(chunk, return_tensors="pt")
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], speaker_id=speaker_id)
        audio_arr = outputs.waveform.squeeze().cpu().numpy()
        audio_segments.append(audio_arr)

    print("Concatenating audio segments...")
    final_audio = np.concatenate(audio_segments)
    output_dir = "audio/output_audio"
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, output_filename)
    sf.write(full_output_path, final_audio, 16000)
    print(f"Generated audio saved at: {full_output_path}")
    return full_output_path


# Integration: ASR → LLM (with RAG) → TTS
def speech_to_speech_pipeline(input_audio, unique_filename: str, speaker_id="EN-US"):
    print("Starting speech-to-speech pipeline...")
    text = speech_to_text(input_audio)
    response = query_groq_llama(text)
    output_audio = text_to_speech(response, output_filename=f"{unique_filename}_output.wav")
    return output_audio


def main(path: str, unique_filename: str):
    output_audio_file = speech_to_speech_pipeline(input_audio=path, speaker_id="EN-US", unique_filename=unique_filename)
    return output_audio_file
