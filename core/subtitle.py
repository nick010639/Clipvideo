import os

def format_timestamp(seconds: float) -> str:
    """Formats a float second value into SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

import re

def count_words(text):
    """Counts words. For CJK, each character is considered a word."""
    # Check if text is primarily CJK (Chinese, Japanese, Korean)
    cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))
    if cjk_chars > len(text) * 0.3: # If more than 30% is CJK
        return cjk_chars
    else:
        return len(text.split())

def split_text_smartly(text, max_words=15, max_cjk_chars=30):
    """
    Splits text into smaller chunks based on punctuation and word/character limits.
    - For English: max 15 words per chunk.
    - For CJK: max 30 characters per chunk.
    Prioritizes splitting at sentence enders (.?!), then commas.
    """
    # First, determine if text is primarily CJK
    cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))
    is_cjk = cjk_chars > len(text) * 0.3
    
    # Calculate effective limit
    if is_cjk:
        effective_limit = max_cjk_chars
    else:
        # For English, roughly 7 chars per word average
        effective_limit = max_words * 7
    
    if len(text) <= effective_limit:
        return [text]
    
    # Split by sentence enders first, keeping delimiters
    parts = [p for p in re.split(r'([。！？.?!，,、；;])', text) if p]
    
    chunks = []
    current_chunk = ""
    
    for item in parts:
        is_punct = re.match(r'^[。！？.?!，,、；;]+$', item.strip())
        
        if is_punct:
            current_chunk += item
            # Check if current chunk exceeds limit - if so, split it
            if is_cjk:
                if len(current_chunk) >= max_cjk_chars:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
            else:
                if len(current_chunk.split()) >= max_words:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
        else:
            # If adding this text would exceed limit, push current first
            test_chunk = current_chunk + item
            if is_cjk:
                exceeds = len(test_chunk) > max_cjk_chars
            else:
                exceeds = len(test_chunk.split()) > max_words
                
            if exceeds and current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = item
            else:
                current_chunk += item
                
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    # Post-processing: Merge very short chunks into previous
    final_chunks = []
    for chunk in chunks:
        if not chunk: continue
        # If chunk is just punctuation, merge with previous
        if len(chunk) < 2 and re.match(r'^[。！？.?!，,]+$', chunk) and final_chunks:
            final_chunks[-1] += chunk
        else:
            final_chunks.append(chunk)

    return final_chunks

def interpolate_timestamps(start, end, text, chunks):
    """
    Distributes time across chunks based on character count.
    """
    total_len = len(text)
    if total_len == 0:
        return []
        
    duration = end - start
    
    results = []
    current_start = start
    
    for i, chunk in enumerate(chunks):
        chunk_len = len(chunk)
        # Calculate proportion of time this chunk takes
        if total_len > 0:
            chunk_duration = (chunk_len / total_len) * duration
        else:
            chunk_duration = duration / len(chunks)
        
        current_end = current_start + chunk_duration
        
        # Adjust last chunk to match exact end time to avoid drift
        if i == len(chunks) - 1:
            current_end = end
            
        results.append({
            "start": current_start,
            "end": current_end,
            "text": chunk
        })
        current_start = current_end
        
    return results

def refine_segments(segments, max_words=15, max_cjk_chars=30, max_duration=8.0, time_offset=0.0):
    """
    Iterates through segments. If a segment is too long (text or duration),
    splits it into smaller sub-segments.
    
    Args:
        segments: List of segment dicts with 'start', 'end', 'text', optional 'speaker'.
        max_words: Max words per chunk for English text.
        max_cjk_chars: Max characters per chunk for CJK text.
        max_duration: Max duration in seconds before forcing a split.
        time_offset: Seconds to add to all timestamps (positive = delay subtitles).
        
    Returns a flattened list of refined segments.
    """
    processed_segments = []
    
    for segment in segments:
        text = segment["text"]
        start = segment["start"] + time_offset
        end = segment["end"] + time_offset
        # Default to empty string if speaker is missing, ensuring it propagates
        speaker = segment.get("speaker", "")
        
        # Threshold check: use word count for splitting decision
        word_count = count_words(text)
        cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))
        is_cjk = cjk_chars > len(text) * 0.3
        
        needs_split = False
        if is_cjk and cjk_chars > max_cjk_chars:
            needs_split = True
        elif not is_cjk and word_count > max_words:
            needs_split = True
        elif (end - start) > max_duration:
            needs_split = True
        
        if needs_split:
            # 1. Split text with new word-based limits
            chunks = split_text_smartly(text, max_words=max_words, max_cjk_chars=max_cjk_chars)
            
            # 2. Interpolate timestamps
            split_segs = interpolate_timestamps(start, end, text, chunks)
            
            # 3. Add speaker info back and append
            for s in split_segs:
                s["speaker"] = speaker
                processed_segments.append(s)
        else:
            # Apply time_offset to non-split segments too
            processed_segments.append({
                "start": start,
                "end": end,
                "text": text,
                "speaker": speaker
            })
            
    return processed_segments

def generate_srt(segments: list, output_path: str):
    """
    Generates an SRT file from transcript segments.
    Expects segments to be already refined/split if necessary.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for index, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            
            text_content = segment["text"]
            
            # User requested NOT to show speaker labels in the SRT text
            # speaker = segment.get("speaker", "")
            # if speaker:
            #      text_content = f"[{speaker}]: {text_content}"

            f.write(f"{index}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text_content}\n\n")
    
    return output_path
