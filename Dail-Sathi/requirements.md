# Dail-Sathi
pip install torch transformers scipy pydub requests numpy soundfile 
# To install IndicTransToolkit
# Clone the github repository and navigate to the project directory.
git clone https://github.com/VarunGumma/IndicTransToolkit
cd IndicTransToolkit
pip install --editable ./
# How To install FFPEG 
For Windows
Method 1: Install via Chocolatey (Recommended)
Open PowerShell (Run as Administrator)
Run the command:
choco install ffmpeg
Restart your terminal.

Method 2: Manual Installation
Download FFmpeg from the official site:
👉 https://ffmpeg.org/download.html
Extract the files and add the bin folder to your System PATH:
Copy the extracted folder to C:\ffmpeg
Go to System Properties > Environment Variables
Under System Variables, find Path, click Edit, and add:

C:\ffmpeg\bin
Restart your terminal and check installation with:

check version with this
ffmpeg -version

# Install Flash Attention 2
Install Dependencies
Before installing FlashAttention-2, ensure you have the required libraries:
pip install torch ninja packaging

Install FlashAttention 2
Run the following command to install it:
pip install flash-attn --no-build-isolation
