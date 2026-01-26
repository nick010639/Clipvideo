import os
import sys
import imageio_ffmpeg
import shutil

# --- Setup FFmpeg (Same as app.py) ---
try:
    ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_cmd)
    
    # Ensure 'ffmpeg.exe' exists in that directory so 'ffmpeg' command works
    ffmpeg_shim = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(ffmpeg_shim):
        shutil.copy2(ffmpeg_cmd, ffmpeg_shim)
    
    os.environ["PATH"] += os.pathsep + ffmpeg_dir
    print(f"FFmpeg path added: {ffmpeg_dir}")
except Exception:
    pass

# Import the actual function to test
sys.path.append(os.getcwd())
try:
    from core.video import burn_subtitles
except ImportError:
    print("Could not import core.video.burn_subtitles")
    sys.exit(1)

# --- Reproduction ---
def create_dummy_files():
    # Create simple video
    if not os.path.exists("dummy_video.mp4"):
        os.system(f'ffmpeg -f lavfi -i color=c=blue:s=320x240:d=5 -c:v libx264 -y dummy_video.mp4')
    
    # Create directory with quote
    complex_dir = "output/Bob's Video"
    os.makedirs(complex_dir, exist_ok=True)
    
    # Use a filename with a quote too
    srt_path = os.path.join(complex_dir, "test's.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("1\n00:00:00,000 --> 00:00:02,000\nHello World\n\n")
    
    output_path = os.path.join(complex_dir, "output's_video.mp4")
    
    return "dummy_video.mp4", srt_path, output_path

if __name__ == "__main__":
    vid, srt, out = create_dummy_files()
    print(f"Testing with SRT: {srt}")
    print(f"Output: {out}")
    
    try:
        burn_subtitles(vid, srt, out)
        print("Success!")
    except Exception as e:
        print(f"Failed with error: {e}")
