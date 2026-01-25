import google.generativeai as genai
import time
import json
import os

class GeminiTranscriber:
    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview"):
        """
        Initializes the Gemini Transcriber.
        """
        if not api_key:
            raise ValueError("API Key is required/API Key 不能为空")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def _upload_to_gemini(self, path, mime_type=None):
        """Uploads the given file to Gemini."""
        print(f"Uploading file '{path}' to Gemini...")
        file = genai.upload_file(path, mime_type=mime_type)
        print(f"Completed upload: {file.uri}")
        return file

    def _wait_for_files_active(self, files):
        """Waits for the given files to be active."""
        print("Waiting for file processing...")
        for name in (file.name for file in files):
            file = genai.get_file(name)
            while file.state.name == "PROCESSING":
                print(".", end="", flush=True)
                time.sleep(2)
                file = genai.get_file(name)
            if file.state.name != "ACTIVE":
                raise Exception(f"File {file.name} failed to process")
        print("...all files ready")

    def transcribe(self, audio_path: str):
        """
        Transcribes audio file to text segments with speaker diarization.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            list: A list of dictionaries containing 'start', 'end', 'speaker', and 'text'.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Upload file (extract_audio produces .mp3 now)
        audio_file = self._upload_to_gemini(audio_path, mime_type="audio/mp3") 
        self._wait_for_files_active([audio_file])

        prompt = (
            "Transcribe the following audio file with PRECISE timestamps. "
            "CRITICAL REQUIREMENTS:\n"
            "1. Each segment must be SHORT - maximum 15 words or one sentence, whichever is shorter.\n"
            "2. Timestamps must be ACCURATE to the actual speech timing in the audio.\n"
            "3. If a speaker talks for a long time, split into multiple short segments with correct timestamps.\n"
            "4. Identify different speakers and label them (e.g., 'Speaker 1', 'Speaker 2').\n"
            "\n"
            "Return a strictly valid JSON list of objects with these keys:\n"
            "- 'start': float, start time in seconds (e.g., 12.5)\n"
            "- 'end': float, end time in seconds (e.g., 15.3)\n"
            "- 'speaker': string, speaker label\n"
            "- 'text': string, the spoken text\n"
            "\n"
            "Example output format:\n"
            "[{\"start\": 0.0, \"end\": 2.5, \"speaker\": \"Speaker 1\", \"text\": \"Hello, how are you?\"}]\n"
            "\n"
            "Do NOT include markdown formatting or backticks. Output ONLY the JSON array."
        )

        print("Requesting transcription from Gemini...")
        response = self.model.generate_content(
            [audio_file, prompt],
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            ),
            request_options={"timeout": 600}
        )

        try:
            # Debugging: Print detailed response info if parts are missing
            if not response.candidates:
                print("Error: No candidates returned from Gemini.")
                return []
            
            candidate = response.candidates[0]
            # Check for safety blocking or other issues
            if not candidate.content.parts:
                print("Error: Gemini returned a candidate with no content parts.")
                print(f"Finish Reason: {candidate.finish_reason}")
                print(f"Safety Ratings: {candidate.safety_ratings}")
                # If blocked by safety, we can't get text
                if candidate.finish_reason == 3: # SAFETY
                    raise Exception("Transcription blocked by safety filters.")
                else:
                    raise Exception(f"Empty response from Gemini. Finish Reason: {candidate.finish_reason}")

            # Clean up response text just in case (though response_mime_type should handle it)
            text_response = response.text.strip()
            if text_response.startswith("```json"):
                text_response = text_response.replace("```json", "").replace("```", "")
            elif text_response.startswith("```"):
                 text_response = text_response.replace("```", "")
            
            segments = json.loads(text_response)
            
            # Basic validation
            validated_segments = []
            for seg in segments:
                if "start" in seg and "end" in seg and "text" in seg:
                    # Ensure speaker key exists
                    if "speaker" not in seg:
                        seg["speaker"] = "Speaker 1"
                    validated_segments.append(seg)
            
            return validated_segments

        except json.JSONDecodeError:
            print(f"JSON Parsing Failed!")
            print(f"Response Length: {len(text_response)}")
            print(f"First 500 chars: {text_response[:500]}")
            print(f"Last 500 chars: {text_response[-500:]}")
            if "finish_reason" in str(response.candidates[0]):
                 print(f"Finish Reason: {response.candidates[0].finish_reason}")
            
            raise RuntimeError(f"Failed to parse Gemini transcription response. Response might be truncated or invalid.")
        except Exception as e:
            print(f"Error during transcription: {e}")
            raise e
