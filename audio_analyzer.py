import sounddevice as sd
import numpy as np
import whisper
from transformers import pipeline
import threading
import queue
import time

class AudioAnalyzer:
    """
    Handles real-time audio recording, transcription, and sentiment analysis.
    """
    def __init__(self, output_queue, sample_rate=16000, record_duration=5, model_size="tiny.en"):
        self.output_queue = output_queue  # The queue to send results to the main app
        self.sample_rate = sample_rate
        self.record_duration = record_duration
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.thread_record = None
        self.thread_transcribe = None

        print("Loading audio analysis models...")
        self.whisper_model = whisper.load_model(model_size)
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        print("Audio analysis models loaded successfully.")

    def _record_audio(self):
        """Continuously records audio and puts it into a local queue."""
        while self.is_running:
            try:
                audio_chunk = sd.rec(int(self.record_duration * self.sample_rate),
                                     samplerate=self.sample_rate, channels=1, dtype='float32')
                sd.wait()
                self.audio_queue.put(audio_chunk)
            except Exception as e:
                print(f"Error during audio recording: {e}")
                time.sleep(1)

    def _transcribe_audio(self):
        """Processes audio, transcribes, analyzes, and puts results in the output_queue."""
        while self.is_running or not self.audio_queue.empty():
            try:
                audio_chunk = self.audio_queue.get(timeout=1)
                audio_np = audio_chunk.flatten()

                result = self.whisper_model.transcribe(audio_np, fp16=False)
                text = result['text'].strip()

                if text:
                    sentiment = self.sentiment_analyzer(text)
                    # Put the final result onto the queue shared with the dashboard
                    self.output_queue.put({
                        'text': text,
                        'sentiment': sentiment[0]
                    })
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error during audio transcription: {e}")

    def start(self):
        """Starts the recording and transcription threads."""
        if self.is_running:
            return
            
        print("Starting audio analyzer...")
        self.is_running = True
        self.thread_record = threading.Thread(target=self._record_audio, daemon=True)
        self.thread_transcribe = threading.Thread(target=self._transcribe_audio, daemon=True)
        self.thread_record.start()
        self.thread_transcribe.start()
        print("Audio analyzer started.")

    def stop(self):
        """Stops the threads."""
        if not self.is_running:
            return
        self.is_running = False
        if self.thread_record: self.thread_record.join()
        if self.thread_transcribe: self.thread_transcribe.join()
        print("Audio analyzer stopped.")
