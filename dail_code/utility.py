import os
import soundfile as sf
import numpy as np


def reprocess_audio_util(input_filename, output_filename="processed.wav"):
    """Reprocess the WAV file to ensure correct formatting using soundfile."""
    output_dir = "./audio/audio_preprocessing/"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    output_path = os.path.join(output_dir, output_filename)

    print(f"Reprocessing audio file: {input_filename}")

    # Read the audio file
    data, samplerate = sf.read(input_filename, dtype="int16")

    # Ensure mono (convert to single channel if needed)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1, dtype="int16")  # Convert stereo to mono

    # Resample if needed
    target_rate = 16000
    if samplerate != target_rate:
        from scipy.signal import resample_poly
        data = resample_poly(data, target_rate, samplerate)
        samplerate = target_rate

    # Save the processed file
    sf.write(output_path, data, samplerate, subtype="PCM_16")

    print(f"Audio reprocessed and saved as {output_path}")
    return output_path
