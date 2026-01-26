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
    
    # Create valid srt
    with open("valid.srt", "w", encoding="utf-8") as f:
        f.write("1\n00:00:00,000 --> 00:00:02,000\nHello World\n\n")
        
    return "dummy_video.mp4", "valid.srt", "non_existent_file.srt", "output_fail.mp4"

if __name__ == "__main__":
    vid, valid_srt, invalid_srt, out = create_dummy_files()
    
    print("\n--- Test 1: Forced Failure (Invalid SRT path) ---")
    try:
        # burn_subtitles handles file existence check, so pass a path that exists but makes ffmpeg fail?
        # Actually burn_subtitles has check: if not os.path.exists(subtitle_path) raise...
        # So let's pass a file that is NOT an SRT to ffmpeg to make it choke?
        # Or inject a bad filter string? We can't easily injection without modifying code.
        # Let's try to corrupt the SRT content after checks?
        
        with open("corrupt.srt", "w") as f:
            f.write("This is garbage content that ffmpeg might dislike")
            
        burn_subtitles(vid, "corrupt.srt", out)
        print("Success (Unexpected for corrupt file)!")
    except Exception as e:
        print(f"Caught expected error: {e}")
        
    print("\n--- Test 2: Valid Call (Should success) ---")
    try:
        burn_subtitles(vid, valid_srt, "output_success.mp4")
        print("Success (Expected)!")
    except Exception as e:
        print(f"Failed (Unexpected): {e}")
