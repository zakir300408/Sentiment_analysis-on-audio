import os
import torch
import whisper
import yt_dlp
import numpy as np
import json
from transformers import pipeline
from pyannote.audio import Pipeline
from pydub import AudioSegment

# Global configurations
PROXY_SETTINGS = {
    'HTTP_PROXY': 'http://127.0.0.1:7890',
    'HTTPS_PROXY': 'http://127.0.0.1:7890'
}
YOUTUBE_URL = "https://www.youtube.com/watch?v=XrmyU3PTCYk"
MODEL_SIZE = "base"
OUTPUT_JSON_FILE = "transcription.json"

# Set proxy environment variables
os.environ.update(PROXY_SETTINGS)
hugging_face_token = os.getenv("hugging_face_token")


def get_audio_filename(video_url):
    """Fetch the YouTube video title and generate a filename based on its first 3 words."""
    ydl_opts = {'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        title = info.get("title", "audio").split()[:3]
        return "_".join(title).replace(" ", "_")


def download_audio_from_youtube(youtube_url, output_filename):
    """Download audio from a YouTube video and save it as an MP3 file."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_filename,
        'quiet': False
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return f"{output_filename}.mp3"


def perform_speaker_diarization(audio_file):
    """
    Perform speaker diarization on the audio file using the updated pyannote.audio API.
    Requires a valid Hugging Face authentication token.
    """
    if not hugging_face_token:
        raise ValueError("Hugging Face token is not set. Please ensure 'hugging_face_token' is initialized.")

    # Load updated diarization model
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hugging_face_token
    )

    # Perform diarization
    diarization = pipeline(audio_file)
    speaker_segments = [
        {"start": segment.start, "end": segment.end, "speaker": label}
        for segment, _, label in diarization.itertracks(yield_label=True)
    ]

    print("Speaker Diarization Completed")
    return speaker_segments


def refine_speaker_segments(speaker_segments):
    """
    Refine speaker segments to handle overlaps and merge adjacent segments.
    """
    refined_segments = []

    for segment in sorted(speaker_segments, key=lambda x: (x["start"], x["end"])):
        if not refined_segments:
            refined_segments.append(segment)
        else:
            last_segment = refined_segments[-1]
            # Merge if the same speaker and the segments overlap or are adjacent
            if (
                last_segment["speaker"] == segment["speaker"]
                and last_segment["end"] >= segment["start"] - 0.1
            ):
                last_segment["end"] = max(last_segment["end"], segment["end"])
            else:
                refined_segments.append(segment)

    return refined_segments


def extract_audio_segments_with_no_overlap(audio_file, speaker_segments):
    """
    Extract audio segments for each speaker while avoiding overlaps.
    """
    refined_segments = refine_speaker_segments(speaker_segments)
    audio = AudioSegment.from_file(audio_file)
    speaker_audio_files = []

    for i, segment in enumerate(refined_segments):
        start_ms = segment["start"] * 1000
        end_ms = segment["end"] * 1000
        segment_audio = audio[start_ms:end_ms]
        segment_file = f"segment_{i + 1}_{segment['speaker']}.wav"
        segment_audio.export(segment_file, format="wav")
        speaker_audio_files.append({
            "file": segment_file,
            "start": segment["start"],
            "end": segment["end"],
            "speaker": segment["speaker"]
        })

    return speaker_audio_files


def transcribe_audio_segments(model, audio_segments):
    """
    Transcribe each audio segment.
    """
    transcriptions = []

    for segment in audio_segments:
        print(f"Processing: {segment['file']} for {segment['speaker']}...")
        # Transcribe the entire audio segment
        result = model.transcribe(
            segment["file"],
            language="en",
            fp16=torch.cuda.is_available()
        )
        transcriptions.append({
            "speaker": segment["speaker"],
            "start": segment["start"],
            "end": segment["end"],
            "text": result["text"]
        })

    return transcriptions


def analyze_sentiment(transcriptions):
    """Analyze sentiment for each transcription using an advanced model."""
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=0 if torch.cuda.is_available() else -1,
    )

    # Map labels to actual sentiment names
    label_mapping = {
        'LABEL_0': 'Negative',
        'LABEL_1': 'Neutral',
        'LABEL_2': 'Positive'
    }

    # Collect all texts
    texts = [t["text"] for t in transcriptions]

    # Analyze sentiments in batch
    sentiments = sentiment_model(texts)

    # Assign sentiments back to transcriptions
    for t, sentiment in zip(transcriptions, sentiments):
        # Map label to sentiment name
        sentiment_label = label_mapping.get(sentiment['label'], sentiment['label'])
        t['sentiment'] = {
            'label': sentiment_label,
            'score': sentiment['score']
        }

    # Overall sentiment per speaker
    speaker_texts = {}
    for t in transcriptions:
        speaker = t["speaker"]
        if speaker not in speaker_texts:
            speaker_texts[speaker] = []
        speaker_texts[speaker].append(t["text"])

    # Prepare texts for each speaker
    speaker_combined_texts = {speaker: " ".join(texts) for speaker, texts in speaker_texts.items()}

    # Analyze overall sentiment for each speaker
    overall_sentiments_list = sentiment_model(list(speaker_combined_texts.values()))

    overall_sentiments = {}
    for speaker, sentiment in zip(speaker_combined_texts.keys(), overall_sentiments_list):
        sentiment_label = label_mapping.get(sentiment['label'], sentiment['label'])
        overall_sentiments[speaker] = {
            'label': sentiment_label,
            'score': sentiment['score']
        }

    return sentiments, overall_sentiments


def save_transcription_to_json(transcriptions, overall_sentiments, output_file):
    """Save transcription and sentiments to JSON."""
    output_data = {
        "transcriptions": transcriptions,
        "overall_sentiments": overall_sentiments
    }
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)
    print(f"Transcription saved to {output_file}")


def main():
    # Get filename based on YouTube title
    audio_filename = get_audio_filename(YOUTUBE_URL)

    # Download audio
    try:
        audio_file = download_audio_from_youtube(YOUTUBE_URL, audio_filename)
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return

    # Perform speaker diarization
    speaker_segments = perform_speaker_diarization(audio_file)

    # Refine and extract audio for each speaker segment
    speaker_audio_files = extract_audio_segments_with_no_overlap(audio_file, speaker_segments)

    # Transcribe audio for each speaker
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(MODEL_SIZE, device=device)

    transcriptions = transcribe_audio_segments(model, speaker_audio_files)

    # Analyze sentiment for each transcription
    segment_sentiments, overall_sentiments = analyze_sentiment(transcriptions)

    # Save transcription and sentiments to JSON
    save_transcription_to_json(transcriptions, overall_sentiments, OUTPUT_JSON_FILE)

    # Clean up temporary audio files
    for segment in speaker_audio_files:
        os.remove(segment["file"])


if __name__ == "__main__":
    main()
