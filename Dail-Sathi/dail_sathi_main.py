from redirect_audio import main
from flask import Flask, request, send_from_directory
from twilio.twiml.voice_response import VoiceResponse
import os
from dotenv import load_dotenv
import time
import sys
import string
import requests
import random
from datetime import datetime

app = Flask(__name__)

load_dotenv()

# Directories for audio files
AUDIO_DIR = './audio'
SELECTED_LANGUAGE_DIR = './audio/Selected_language'
CALLER_RECORDINGS_DIR = './audio/caller_recorded_audio'

# Twilio Credentials (Replace with your actual credentials)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# Ensure directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(SELECTED_LANGUAGE_DIR, exist_ok=True)
os.makedirs(CALLER_RECORDINGS_DIR, exist_ok=True)


def generate_unique_filename(length=10):
    # Generate a random string of uppercase, lowercase letters, and digits
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    unique_str_10 = f"{random_string}"

    return unique_str_10

@app.route('/audio/<filename>', methods=['GET'])
def serve_audio(filename):
    """Serve audio files from the root audio directory."""
    try:
        print(f"Playing Intro-audio-file: {filename}")
        return send_from_directory(AUDIO_DIR, filename)
    except Exception as e:
        return f"Error serving file {filename}: {str(e)}", 500

@app.route('/audio/Selected_language/<filename>', methods=['GET'])
def serve_selected_language_audio(filename):
    """Serve selected language audio files."""
    try:
        print(f"Playing Selected_Language-audio-file: {filename}")
        return send_from_directory(SELECTED_LANGUAGE_DIR, filename)
    except Exception as e:
        return f"Error serving file {filename}: {str(e)}", 500

@app.route('/audio/caller_recorded_audio/<filename>', methods=['GET'])
def serve_caller_recorded_audio(filename):
    """Serve caller-recorded audio files."""
    try:
        print(f"Caller-recorded-audio-file: {filename}")
        return send_from_directory(CALLER_RECORDINGS_DIR, filename)
    except Exception as e:
        return f"Error serving file {filename}: {str(e)}", 500

@app.route("/voice", methods=["GET", "POST"])
def voice():
    """Initial greeting and language selection."""
    response = VoiceResponse()
    try:
        audio_url = request.url_root + 'audio/Final_intro_v1.wav'
        response.play(audio_url)
        response.say("Beep")
        response.gather(
            num_digits=1,
            action='/selected_language',
            method='POST',
            input='dtmf'
        )
    except Exception as e:
        response.say("An error occurred. Please try again later.")
        print(f"Error in /voice: {str(e)}")

    return str(response)

@app.route('/selected_language', methods=['POST'])
def language():
    """Handle selected_language selection and initiate recording."""
    response = VoiceResponse()
    try:
        selected_option = request.form.get('Digits')
        language_map = {
            '1': 'english',
            '2': 'hindi',
            '3': 'kannada',
            '4': 'tamil',
            '5': 'telugu',
            '6': 'malayalam'
        }

        if selected_option in language_map:
            selected_language = language_map[selected_option]
            audio_filename = f"{selected_language}_selected_v1.wav"
            audio_url = request.url_root + f'audio/Selected_language/{audio_filename}'
            response.play(audio_url)
            print(f'URL Root: {request.url_root}\nAudio URL: {audio_url}')

            # Generate a unique filename
            timestamp = int(time.time())
            formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')
            recording_filename = f"{str(generate_unique_filename())}_{selected_language[:2]}"

            # Pass filename via query parameters to avoid session issues
            response.record(
                max_length=10,
                play_beep=True,
                recordingStatusCallback=f'/save_recording?filename={recording_filename + "_input.wav"}&selected_option={selected_option}',
                recording_format='wav',
                action=f'/dailSathiAI?input_audio_file={recording_filename + "_input.wav"}&selected_langauge={selected_option}&unique_filename={recording_filename}'
            )
            response.say("Thank you for calling Dial Sathi!")
        else:
            response.say("Invalid selection. Please try again.")
            response.redirect('/voice')

    except Exception as e:
        response.say("An error occurred while processing your request.")
        print(f"Error in /selected_language: {str(e)}")

    return str(response)

@app.route('/dailSathiAI', methods=['POST'])
def dail_sathi_ai():
    """Waits for a few seconds and then plays back the recorded audio."""
    response = VoiceResponse()

    # Print a 10-second countdown before proceeding
    for i in range(10, 0, -1):
        print(f"Waiting... {i} seconds remaining", end="\r")
        sys.stdout.flush()
        time.sleep(1)

    print("\nProceeding with execution...")

    input_audio_file = request.args.get('input_audio_file')
    selected_language = request.args.get('selected_langauge')  # FIXED: Corrected the spelling
    unique_filename = request.args.get('unique_filename')

    # Validate required parameters
    if not input_audio_file or not selected_language or not unique_filename:
        print(f"Error: Missing parameters. Received - input_audio_file: {input_audio_file}, selected_language: {selected_language}, unique_filename: {unique_filename}")
        response.say("Error processing your request due to missing parameters.")
        return str(response)

    print(f"Input Audio File: {input_audio_file}")
    print(f"Selected Language: {selected_language}")
    print(f"Unique Filename: {unique_filename}")

    input_file_path = os.path.join(CALLER_RECORDINGS_DIR, input_audio_file)

    if os.path.exists(input_file_path):
        try:
            output_audio_path = main(
                path=f"audio/caller_recorded_audio/{input_audio_file}",
                language=str(selected_language),  # Ensuring it's a string
                unique_filename=unique_filename
            )

            if output_audio_path and os.path.exists(output_audio_path):
                response.play(request.url_root + output_audio_path)
            else:
                print("Error: Output audio file does not exist.")
                response.say("Output audio file does not exist.")

        except Exception as e:
            print(f"Error in main(): {str(e)}")
            response.say("An error occurred while processing your request.")

    else:
        print("Error: Input audio file does not exist.")
        response.say("Input audio file does not exist.")

    return str(response)

@app.route('/save_recording', methods=['POST'])
def save_recording():
    """Download and save the recorded caller audio."""
    response_msg = VoiceResponse()

    if request.args.get('selected_option') != "timeout":
        try:
            recording_url = request.form.get("RecordingUrl")  # Get Twilio recording URL
            filename = request.args.get('filename')  # Read filename from query params
            print(f"Entered /save_recording: {filename}, {recording_url}")

            if not recording_url:
                print("Recording URL not received.")
                return str(response_msg)

            try:
                # Authenticate with Twilio when downloading the recording
                response = requests.get(recording_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
                response.raise_for_status()  # Raise an error for failed HTTP requests
                print("Successfully downloaded the recording.")
            except requests.exceptions.RequestException as e:
                print(f"Failed to download recording: {str(e)}")
                return str(response_msg)

            # Save the audio file locally
            file_path = os.path.join(CALLER_RECORDINGS_DIR, filename)
            try:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Recording saved as {file_path}")
            except IOError as e:
                print(f"Error writing file {file_path}: {str(e)}")
                return str(response_msg)

        except Exception as e:
            print(f"Error in /save_recording: {str(e)}")

    else:
        print(f"Selected option: {request.args.get('selected_option')}")

    return str(response_msg)

if __name__ == "__main__":
    app.run(debug=False)
