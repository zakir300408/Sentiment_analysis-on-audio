
# Sentiment Analysis on Audio

This project performs sentiment analysis on audio extracted from YouTube videos. It downloads audio from a YouTube video, performs speaker diarization to separate different speakers, transcribes the audio for each speaker, and analyzes the sentiment of the transcriptions.

## Overview

Sentiment analysis on audio involves processing audio recordings to detect the speaker's emotions. This can be useful in various applications such as customer service, mental health monitoring, and more.

## Requirements

To run this script, you need to install the following dependencies:

- `torch`
- `whisper`
- `yt_dlp`
- `numpy`
- `transformers`
- `pyannote.audio`
- `pydub`

Additionally, you need to set up a Hugging Face authentication token in your environment:


```sh
export hugging_face_token=<YOUR_HUGGING_FACE_TOKEN>
```

1. Clone the repository:
   ```sh
   git clone https://github.com/zakir300408/Sentiment_analysis-on-audio.git
   cd Sentiment_analysis-on-audio
   ```
## Usage
Update the YOUTUBE_URL variable in the script with the URL of the YouTube video you want to analyze.
The script will perform the following steps:

Download the audio from the specified YouTube video.
Perform speaker diarization to identify different speakers in the audio.
Extract audio segments for each speaker while avoiding overlaps.
Transcribe the audio segments for each speaker.
Analyze the sentiment of each transcription.
Save the transcriptions and sentiment analysis results to a JSON file (transcription.json).


