import torch
from transformers import pipeline
import whisper

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load Whisper model and transcribe audio on the specified device (GPU or CPU)
model = whisper.load_model("small", device=device)  # Load model to GPU or CPU
audio = whisper.load_audio("eng.mp3")
audio = whisper.pad_or_trim(audio)

# Make the transcription (without translation)
result = model.transcribe(audio, language="en", fp16=torch.cuda.is_available())

# Output the transcription (in English)
transcription = result["text"]
print("Transcription (English): ", transcription)

# Step 2: Zero-shot classification using Hugging Face model (BART-large-MNLI), ensuring it's on the specified device (GPU or CPU)
classification_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

# Perform zero-shot classification on the English transcription
classification_result = classification_model(transcription, candidate_labels=["positive", "negative", "neutral"])

# Output classification result (on the English transcription)
print("Zero-shot Classification Result (English Transcription): ", classification_result)

# Step 3: Sentiment Analysis using the same BART model
sentiment_model = pipeline("sentiment-analysis", model="facebook/bart-large-mnli", device=device)

# Perform sentiment analysis on the English transcription
sentiment_result = sentiment_model(transcription)

# Output sentiment result (on the English transcription)
print("Sentiment Analysis Result (English Transcription): ", sentiment_result)
