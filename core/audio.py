import ffmpeg
import os

def extract_audio(video_path: str, output_path: str = None) -> str:
    """
    Extracts audio from a video file and saves it as a WAV file.
    
    Args:
        video_path (str): Path to the input video file.
        output_path (str, optional): Path for the output audio file. 
                                     If None, it defaults to the same name as video but with .wav extension.
    
    Returns:
        str: Path to the generated audio file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        base, _ = os.path.splitext(video_path)
        output_path = f"{base}.mp3"

    try:
        # Extract audio: mono, 16kHz, 64k bitrate MP3 is optimized for speech
        # -y overwrites output file if it exists
        (
            ffmpeg
            .input(video_path)
            .output(output_path, ac=1, ar='16000', audio_bitrate='64k')
            .overwrite_output()
            .run(quiet=True)
        )
        return output_path
    except ffmpeg.Error as e:
        print(f"An error occurred: {e.stderr.decode('utf8')}")
        raise
