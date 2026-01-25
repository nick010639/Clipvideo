import google.generativeai as genai
import time
import json

class GeminiTranslator:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize Gemini Translator with API Key.
        """
        if not api_key:
            raise ValueError("API Key is required/API Key 不能为空")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def translate_segments(self, segments: list, batch_size: int = 20) -> list:
        """
        Translates a list of subtitle segments from English to Chinese.
        
        Args:
            segments (list): List of dicts with 'text'.
            batch_size (int): Number of segments to translate in one batch to respect limits and context.
            
        Returns:
            list: The input list with a new 'text_zh' key for each segment.
        """
        translated_segments = segments.copy()
        total_segments = len(segments)
        
        print(f"Starting translation for {total_segments} segments...")

        for i in range(0, total_segments, batch_size):
            batch = segments[i : i + batch_size]
            batch_texts = [seg['text'] for seg in batch]
            
            # Construct Prompt
            numbered_text = "\n".join([f"{idx+1}. {text}" for idx, text in enumerate(batch_texts)])
            prompt = (
                "You are a professional subtitle translator. Translate the following English subtitle lines into Simplified Chinese.\n"
                "Requirements:\n"
                "1. Maintain the context and tone.\n"
                "2. Return EXACTLY one translation line for each input line.\n"
                "3. Do not merge or split lines.\n"
                "4. Output format: A JSON list of strings. Example: [\"你好\", \"世界\"]\n\n"
                "Input:\n"
                f"{numbered_text}"
            )

            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        response_mime_type="application/json"
                    ),
                    request_options={"timeout": 600}
                )
                
                # Simple parsing logic (expecting JSON)
                try:
                    text_response = response.text.strip()
                    # With json mode, clean up might not be needed as much but safe to keep basic
                    if text_response.startswith("```json"):
                        text_response = text_response.replace("```json", "").replace("```", "")
                    elif text_response.startswith("```"):
                         text_response = text_response.replace("```", "")
                    
                    translations = json.loads(text_response)
                    
                    if len(translations) != len(batch):
                        print(f"Warning: Batch {i//batch_size + 1} mismatch. Input: {len(batch)}, Output: {len(translations)}. fallback to manual alignment.")
                        # Fallback or just align as much as possible
                    
                        # Fallback or just align as much as possible
                    
                    for j, translation in enumerate(translations):
                        if j < len(batch):
                            translated_segments[i + j]['text_zh'] = translation
                            if "speaker" in batch[j]:
                                translated_segments[i + j]['speaker'] = batch[j]['speaker']
                            
                except json.JSONDecodeError:
                    print(f"Error parsing JSON for batch {i}. Raw response: {response.text}")
                    # Fallback logic could go here, for now just copy original
                    for j in range(len(batch)):
                        translated_segments[i + j]['text_zh'] = "[Trans Error] " + batch[j]['text']

            except Exception as e:
                print(f"API Error during translation: {e}")
                for j in range(len(batch)):
                    translated_segments[i + j]['text_zh'] = f"[API Error: {str(e)}] " + batch[j]['text']
            
            # Sleep briefly to avoid rate limits if necessary (Flash is high rate limit usually)
            time.sleep(0.5)

        return translated_segments
