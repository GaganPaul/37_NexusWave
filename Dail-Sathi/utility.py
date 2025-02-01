import os
from pydub import AudioSegment

def reprocess_audio_util(input_filename, output_filename="processed.wav"):
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