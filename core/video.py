import ffmpeg
import os

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
    
    s_path = subtitle_path.replace("\\", "/").replace(":", "\\:")
    
    try:
        print(f"Burning subtitles: {subtitle_path} into {video_path}")
        (
            ffmpeg
            .input(video_path)
            .output(output_path, vf=f"subtitles='{s_path}'", acodec='copy')
            .overwrite_output()
            .run(quiet=False)
        )
        return output_path
    except ffmpeg.Error as e:
        print(f"Error burning subtitles: {e.stderr.decode('utf8') if e.stderr else str(e)}")
        raise
