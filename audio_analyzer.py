import sounddevice as sd
import numpy as np
import whisper
from transformers import pipeline
import threading
import queue
import time

class AudioAnalyzer:
    """
    Handles real-time audio recording, transcription, and sentiment analysis in background threads.
    """
    def __init__(self, sample_rate=16000, record_duration=5, model_size="tiny.en"):
        self.sample_rate = sample_rate
        self.record_duration = record_duration
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.is_running = False
        self.thread_record = None
        self.thread_transcribe = None

        print("Loading audio analysis models...")
        # Load OpenAI Whisper model for speech-to-text
        self.whisper_model = whisper.load_model(model_size)
        # Load Hugging Face pipeline for sentiment analysis
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        print("Audio analysis models loaded successfully.")

    def _record_audio(self):
        """Continuously records audio in chunks and puts them in a queue. Runs in a thread."""
        while self.is_running:
            try:
                audio_chunk = sd.rec(int(self.record_duration * self.sample_rate),
                                     samplerate=self.sample_rate, channels=1, dtype='float32')
                sd.wait()  # Wait for the recording to complete
                self.audio_queue.put(audio_chunk)
            except Exception as e:
                print(f"Error during audio recording: {e}")
                time.sleep(1)

    def _transcribe_audio(self):
        """Continuously processes audio from the queue, transcribes it, and analyzes sentiment. Runs in a thread."""
        while self.is_running or not self.audio_queue.empty():
            try:
                audio_chunk = self.audio_queue.get(timeout=1)
                audio_np = audio_chunk.flatten()

                # Transcribe using Whisper
                result = self.whisper_model.transcribe(audio_np, fp16=False) # fp16=False for CPU
                text = result['text'].strip()

                if text:  # If transcription is not empty
                    # Perform sentiment analysis
                    sentiment = self.sentiment_analyzer(text)
                    self.transcription_queue.put({
                        'text': text,
                        'sentiment': sentiment[0] # The result is a list containing one dict
                    })
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error during audio transcription: {e}")

    def start(self):
        """Starts the recording and transcription threads."""
        if self.is_running:
            print("Audio analyzer is already running.")
            return
            
        print("Starting audio analyzer...")
        self.is_running = True
        self.thread_record = threading.Thread(target=self._record_audio, daemon=True)
        self.thread_transcribe = threading.Thread(target=self._transcribe_audio, daemon=True)
        self.thread_record.start()
        self.thread_transcribe.start()
        print("Audio analyzer started.")

    def stop(self):
        """Stops the recording and transcription threads."""
        if not self.is_running:
            return
            
        print("Stopping audio analyzer...")
        self.is_running = False
        if self.thread_record:
            self.thread_record.join()
        if self.thread_transcribe:
            self.thread_transcribe.join()
        print("Audio analyzer stopped.")

    def get_latest_transcription(self):
        """Returns the latest transcription result from the queue without blocking."""
        try:
            return self.transcription_queue.get_nowait()
        except queue.Empty:
            return None
