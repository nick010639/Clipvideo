import ffmpeg
import os
import re

def sanitize_filename(filename: str) -> str:
    """
    Removes or replaces problematic characters from filename for FFmpeg compatibility.
    Keeps the path structure intact, only sanitizes the base filename.
    """
    directory = os.path.dirname(filename)
    basename = os.path.basename(filename)
    
    # Replace problematic characters with safe alternatives
    # Single quotes, double quotes, and other special chars that break FFmpeg filters
    sanitized = re.sub(r"['\"]", "", basename)  # Remove quotes
    sanitized = re.sub(r"[<>:\"|?*]", "_", sanitized)  # Replace Windows-invalid chars
    
    return os.path.join(directory, sanitized)

def burn_subtitles(video_path: str, subtitle_path: str, output_path: str):
    """
    Burns subtitles into the video using FFmpeg.
    
    Args:
        video_path (str): Path to the input video.
        subtitle_path (str): Path to the subtitle file (.srt).
        output_path (str): Path to save the video with burned subtitles.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(subtitle_path):
        raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # FFmpeg requires special escaping for Windows paths in filter
    # Convert backslashes to forward slashes and escape colon
    # Alternative: Use relative path if possible, but absolute path is safer if formatted correctly
    # The 'subtitles' filter syntax: subtitles='filename'
    
    # We will try to use relative paths to avoid the drive letter colon mess if possible
    # But usually absolute path with forward slashes works well if we handle the drive letter
    
    # Easiest way avoids the drive letter issue entirely by running ffmpeg from the dir or using relative
    # But let's try standard forward slash conversion first.
    
    # On Windows, ffmpeg path escaping for filters: 
    # 'D:/folder/file.srt' -> 'D\\:/folder/file.srt' might be needed
    # actually: "subtitles='D\:/a/b.srt'"
    
    # Simple replace for forward slashes usually works, but colon needs escaping for the filter string parsing
    # Also escape single quotes because the filter argument is wrapped in single quotes
    s_path = subtitle_path.replace("\\", "/").replace(":", "\\:").replace("'", r"\'")
    
    try:
        print(f"Burning subtitles: {subtitle_path} into {video_path}")
        (
            ffmpeg
            .input(video_path)
            .output(output_path, vf=f"subtitles='{s_path}'", acodec='copy')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return output_path
    except ffmpeg.Error as e:
        # Capture stderr which contains the actual error message
        error_message = "Unknown FFmpeg error"
        if e.stderr:
            try:
                # Try UTF-8 first
                error_message = e.stderr.decode('utf8')
            except UnicodeDecodeError:
                try:
                    # Try Windows Default (GBK/CP936)
                    error_message = e.stderr.decode('cp936', errors='ignore')
                except Exception:
                    error_message = str(e.stderr)
        
        print(f"Error burning subtitles: {error_message}")
        # Raise a RuntimeError with the detailed message so the UI can show it
        raise RuntimeError(f"FFmpeg failed with error:\n{error_message}") from e
