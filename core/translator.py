import json
import os
import re
import time
import urllib.error
import urllib.request

import anthropic


class MiniMaxTranslator:
    def __init__(
        self,
        api_key: str,
        model_name: str = "MiniMax-M2.7",
        base_url: str = "https://api.minimaxi.com/anthropic",
    ):
        if not api_key:
            raise ValueError("MiniMax API Key is required.")

        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    def _extract_text(self, response) -> str:
        text_blocks = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        return "\n".join(text_blocks).strip()

    def translate_segments(self, segments: list, batch_size: int = 20) -> list:
        translated_segments = [dict(segment) for segment in segments]
        total_segments = len(segments)

        print(f"Starting translation for {total_segments} segments with {self.model_name}...")

        system_prompt = (
            "You are a professional subtitle translator. "
            "Translate English subtitle lines into natural, concise Simplified Chinese."
        )

        for i in range(0, total_segments, batch_size):
            batch = segments[i : i + batch_size]
            numbered_text = "\n".join(f"{idx + 1}. {seg['text']}" for idx, seg in enumerate(batch))
            prompt = (
                "Translate the following subtitle lines.\n"
                "Requirements:\n"
                "1. Preserve tone and context.\n"
                "2. Return exactly one Chinese translation for each input line.\n"
                "3. Do not merge lines, split lines, add explanations, or add numbering.\n"
                "4. Output only a valid JSON array of strings.\n"
                'Example: ["ni hao", "shi jie"]\n\n'
                "Input:\n"
                f"{numbered_text}"
            )

            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=2048,
                    system=system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ],
                )

                text_response = self._extract_text(response)
                if text_response.startswith("```json"):
                    text_response = text_response.replace("```json", "", 1).replace("```", "").strip()
                elif text_response.startswith("```"):
                    text_response = text_response.replace("```", "").strip()

                translations = json.loads(text_response)
                if not isinstance(translations, list):
                    raise ValueError("Model did not return a JSON array.")

                self._merge_batch_results(translated_segments, batch, translations, i)
            except Exception as e:
                print(f"MiniMax API error during translation: {e}")
                self._mark_batch_error(translated_segments, batch, i, f"[API Error: {str(e)}]")

            time.sleep(0.5)

        return translated_segments

    def _merge_batch_results(self, translated_segments, batch, translations, offset):
        if len(translations) != len(batch):
            print(
                f"Warning: batch mismatch. Input: {len(batch)}, Output: {len(translations)}."
            )

        for j, original_segment in enumerate(batch):
            if j < len(translations) and isinstance(translations[j], str):
                translated_segments[offset + j]["text_zh"] = translations[j]
            else:
                translated_segments[offset + j]["text_zh"] = "[Trans Error] " + original_segment["text"]

            if "speaker" in original_segment:
                translated_segments[offset + j]["speaker"] = original_segment["speaker"]

    def _mark_batch_error(self, translated_segments, batch, offset, prefix):
        for j, original_segment in enumerate(batch):
            translated_segments[offset + j]["text_zh"] = f"{prefix} {original_segment['text']}"


class OllamaTranslator:
    def __init__(
        self,
        model_name: str = "qwen3.5:9b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 180,
        default_batch_size: int = 1,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
        think: bool = False,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_batch_size = max(1, int(default_batch_size))
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.think = bool(think)

    def _post_json(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Ollama HTTP {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to Ollama at {url}: {e.reason}") from e

        return json.loads(body)

    def _extract_response_text(self, response: dict) -> str:
        text_response = response.get("response", "")
        if isinstance(text_response, str) and text_response.strip():
            return text_response.strip()

        def _coerce_content(value) -> str:
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, list):
                parts = []
                for item in value:
                    if isinstance(item, str):
                        if item.strip():
                            parts.append(item.strip())
                    elif isinstance(item, dict):
                        text_part = item.get("text")
                        if isinstance(text_part, str) and text_part.strip():
                            parts.append(text_part.strip())
                        content_part = item.get("content")
                        if isinstance(content_part, str) and content_part.strip():
                            parts.append(content_part.strip())
                return "\n".join(parts).strip()
            return ""

        message = response.get("message") or {}
        if isinstance(message, dict):
            content = _coerce_content(message.get("content"))
            if content:
                return content

        content = _coerce_content(response.get("content"))
        if content:
            return content

        return ""

    def _parse_json_array(self, text_response: str):
        cleaned = text_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "", 1).replace("```", "").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        idx = 0
        parsed_arrays = []
        while idx < len(cleaned):
            next_bracket = cleaned.find("[", idx)
            if next_bracket == -1:
                break
            try:
                candidate, end = decoder.raw_decode(cleaned[next_bracket:])
            except json.JSONDecodeError:
                idx = next_bracket + 1
                continue

            if isinstance(candidate, list):
                parsed_arrays.append(candidate)
            idx = next_bracket + end

        if parsed_arrays:
            if len(parsed_arrays) == 1:
                return parsed_arrays[0]
            merged = []
            for candidate in parsed_arrays:
                merged.extend(candidate)
            return merged

        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict) and "translations" in obj:
                return obj["translations"]
            raise
        except json.JSONDecodeError:
            matches = re.findall(r"\[[\s\S]*?\]", cleaned)
            for candidate in matches:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
            raise

    def _parse_translation_lines(self, text_response: str) -> list:
        cleaned = text_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "", 1).replace("```", "").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "").strip()

        lines = []
        for raw_line in cleaned.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            if line.startswith("- "):
                line = line[2:].strip()

            if line.startswith('"') and line.endswith('"'):
                try:
                    line = json.loads(line)
                except json.JSONDecodeError:
                    line = line.strip('"')

            line = line.rstrip(",").strip()
            if line:
                lines.append(line)

        return lines

    def _clean_single_translation(self, text_response: str) -> str:
        cleaned = text_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "", 1).replace("```", "").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "").strip()

        cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned)
        cleaned = cleaned.rstrip(",").strip()

        if cleaned.startswith('"') and cleaned.endswith('"'):
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, str):
                    return parsed.strip()
            except json.JSONDecodeError:
                cleaned = cleaned.strip('"')

        return cleaned

    def _coerce_translations(self, text_response: str, expected_count: int):
        if expected_count == 1:
            try:
                translations = self._parse_json_array(text_response)
            except json.JSONDecodeError:
                single_text = self._clean_single_translation(text_response)
                return [single_text] if single_text else []

            if isinstance(translations, list) and len(translations) == 1 and isinstance(translations[0], str):
                text = translations[0].strip()
                return [text] if text else []

            single_text = self._clean_single_translation(text_response)
            return [single_text] if single_text else []

        try:
            translations = self._parse_json_array(text_response)
        except json.JSONDecodeError:
            translations = self._parse_translation_lines(text_response)

        if isinstance(translations, dict) and "translations" in translations:
            translations = translations["translations"]
        if not isinstance(translations, list):
            raise ValueError("Model did not return a JSON array.")

        normalized = []
        for item in translations:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    normalized.append(text)
            elif item is not None:
                normalized.append(str(item).strip())
        return normalized

    def _build_prompt(self, batch: list) -> str:
        if len(batch) == 1:
            return (
                "You are a professional subtitle translator.\n"
                "Translate the following English subtitle line into natural Simplified Chinese.\n"
                "Rules:\n"
                "1. Keep the original meaning and tone.\n"
                "2. Output only the Chinese translation text.\n"
                "3. Do not add quotes, numbering, explanations, or extra lines.\n\n"
                "Input:\n"
                f"{batch[0]['text']}"
            )

        numbered_text = "\n".join(f"{idx + 1}. {seg['text']}" for idx, seg in enumerate(batch))
        return (
            "You are a professional subtitle translator.\n"
            "Translate the following English subtitle lines into natural Simplified Chinese.\n"
            "Rules:\n"
            "1. Keep the original meaning, tone, and order.\n"
            "2. Return exactly one Chinese translation for each input line.\n"
            "3. Do not add explanations or line numbers.\n"
            "4. Output only valid JSON.\n"
            '5. The final output must be a JSON array of strings like ["ni hao", "shi jie"].\n\n'
            "Input:\n"
            f"{numbered_text}"
        )

    def _translate_batch_once(self, batch: list) -> list:
        prompt = self._build_prompt(batch)
        expect_json_array = len(batch) > 1
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": self.think,
            "options": {
                "temperature": 0.2,
                "num_ctx": 2048,
                "num_predict": 1024,
            },
        }
        if expect_json_array:
            payload["format"] = "json"

        response = self._post_json("/api/chat", payload)
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(f"Ollama error: {response['error']}")

        text_response = self._extract_response_text(response)
        if not text_response:
            generate_payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "think": self.think,
                "options": {
                    "temperature": 0.2,
                    "num_ctx": 2048,
                    "num_predict": 1024,
                },
            }
            if expect_json_array:
                generate_payload["format"] = "json"
            fallback_response = self._post_json("/api/generate", generate_payload)
            if isinstance(fallback_response, dict) and fallback_response.get("error"):
                raise RuntimeError(f"Ollama error: {fallback_response['error']}")
            text_response = self._extract_response_text(fallback_response)

        if not text_response:
            done_reason = response.get("done_reason", "unknown") if isinstance(response, dict) else "unknown"
            raise ValueError(f"Empty response from Ollama (done_reason={done_reason}).")

        translations = self._coerce_translations(text_response, len(batch))

        if len(translations) != len(batch):
            print(f"Warning: batch mismatch. Input: {len(batch)}, Output: {len(translations)}.")
            if len(batch) > 1:
                raise ValueError(f"Batch mismatch. Input: {len(batch)}, Output: {len(translations)}.")

        normalized = []
        for j, original_segment in enumerate(batch):
            if j < len(translations) and isinstance(translations[j], str):
                normalized.append(translations[j])
            else:
                normalized.append("[Trans Error] " + original_segment["text"])
        return normalized

    def _is_retryable_ollama_error(self, error: Exception) -> bool:
        text = str(error).lower()
        retryable_markers = [
            "ollama http 500",
            "model runner has unexpectedly stopped",
            "resource limitations",
            "timed out",
            "connection reset",
            "remote end closed connection",
            "failed to connect",
            "empty response",
        ]
        return any(marker in text for marker in retryable_markers)

    def _translate_batch_resilient(self, batch: list) -> list:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._translate_batch_once(batch)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries and self._is_retryable_ollama_error(e):
                    wait_seconds = self.retry_backoff_seconds * (attempt + 1)
                    if wait_seconds > 0:
                        time.sleep(wait_seconds)
                    continue
                break

        if len(batch) > 1:
            split_index = max(1, len(batch) // 2)
            print(
                f"Batch of {len(batch)} failed ({last_error}). "
                f"Retrying with smaller chunks: {split_index}+{len(batch) - split_index}."
            )
            left = self._translate_batch_resilient(batch[:split_index])
            right = self._translate_batch_resilient(batch[split_index:])
            return left + right

        raise last_error

    def translate_segments(self, segments: list, batch_size: int = None) -> list:
        translated_segments = [dict(segment) for segment in segments]
        total_segments = len(segments)
        effective_batch_size = max(1, int(batch_size or self.default_batch_size))

        print(f"Starting translation for {total_segments} segments with Ollama model {self.model_name}...")

        for i in range(0, total_segments, effective_batch_size):
            batch = segments[i : i + effective_batch_size]
            try:
                translations = self._translate_batch_resilient(batch)
                for j, original_segment in enumerate(batch):
                    translated_segments[i + j]["text_zh"] = translations[j]
                    if "speaker" in original_segment:
                        translated_segments[i + j]["speaker"] = original_segment["speaker"]
            except Exception as e:
                print(f"Ollama API error during translation: {e}")
                for j, original_segment in enumerate(batch):
                    translated_segments[i + j]["text_zh"] = f"[API Error: {str(e)}] {original_segment['text']}"

        return translated_segments


def _safe_int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _safe_float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _safe_bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def build_translator(
    provider: str,
    model_name: str,
    api_key: str = "",
    base_url: str = "",
):
    if provider == "ollama":
        return OllamaTranslator(
            model_name=model_name or "qwen3.5:9b",
            base_url=base_url or "http://127.0.0.1:11434",
            timeout=_safe_int_from_env("OLLAMA_TIMEOUT", 180),
            default_batch_size=_safe_int_from_env("OLLAMA_BATCH_SIZE", 1),
            max_retries=_safe_int_from_env("OLLAMA_MAX_RETRIES", 2),
            retry_backoff_seconds=_safe_float_from_env("OLLAMA_RETRY_BACKOFF", 1.5),
            think=_safe_bool_from_env("OLLAMA_THINK", False),
        )

    if provider == "minimax":
        return MiniMaxTranslator(
            api_key=api_key,
            model_name=model_name or "MiniMax-M2.7",
            base_url=base_url or "https://api.minimaxi.com/anthropic",
        )

    raise ValueError(f"Unsupported translation provider: {provider}")
