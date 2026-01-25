from faster_whisper import WhisperModel
import os

class Transcriber:
    def __init__(self, model_size="medium", device="cuda", compute_type="float16"):
        """
        Initializes the Whisper model.
        
        Args:
            model_size (str): Size of the Whisper model (tiny, base, small, medium, large-v3).
            device (str): Device to run on ('cuda' or 'cpu').
            compute_type (str): Type of quantization ('float16', 'int8_float16', 'int8').
        """
        print(f"Loading Whisper model '{model_size}' on {device}...")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            error_msg = str(e)
            if "Library cublas" in error_msg or "Library cudnn" in error_msg:
                print(f"CUDA libraries missing: {error_msg}")
                if device == "cuda":
                    print("Attempting fallback to CPU...")
                    try:
                        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
                        print("Fallback to CPU successful.")
                        # You might want to notify the UI somehow, but print is a start
                        return
                    except Exception as cpu_e:
                        raise RuntimeError(f"Failed to initialize Whisper on CUDA (missing libs) and CPU fallback failed: {cpu_e}")
            
            print(f"Failed to load model on {device}: {e}")
            raise e

    def transcribe(self, audio_path: str):
        """
        Transcribes audio file to text segments.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            list: A list of dictionaries containing 'start', 'end', and 'text'.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print("Starting transcription...")
        # Enable VAD (Voice Activity Detection) to skip silence and get accurate start times
        # Enable word_timestamps for more precise segment boundaries
        segments, info = self.model.transcribe(
            audio_path, 
            beam_size=5,
            vad_filter=True,           # Skip silence, detect actual speech
            vad_parameters=dict(
                min_silence_duration_ms=500,  # Minimum silence to split
                speech_pad_ms=200,            # Padding around speech
            ),
            word_timestamps=True       # Get word-level timestamps for accuracy
        )
        
        print(f"Detected language '{info.language}' with probability {info.language_probability}")

        result_segments = []
        segment_count = 0
        for segment in segments:
            segment_count += 1
            if segment_count % 10 == 0:
                print(f"Processed {segment_count} segments...")
            result_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
        
        print(f"Transcription complete: {len(result_segments)} segments.")
        return result_segments
