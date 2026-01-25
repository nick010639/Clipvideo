import streamlit as st
import os

# --- FFmpeg Path Fix ---
# Add the bundled ffmpeg binary to system path so ffmpeg-python can find it
try:
    import imageio_ffmpeg
    import shutil
    ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_cmd)
    
    # Ensure 'ffmpeg.exe' exists in that directory so 'ffmpeg' command works
    ffmpeg_shim = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(ffmpeg_shim):
        shutil.copy2(ffmpeg_cmd, ffmpeg_shim)
        print(f"Created ffmpeg.exe shim at {ffmpeg_shim}")

    os.environ["PATH"] += os.pathsep + ffmpeg_dir
    print(f"FFmpeg path added: {ffmpeg_dir}")
except ImportError:
    print("imageio-ffmpeg not found. Assuming ffmpeg is in system PATH.")
except Exception as e:
    print(f"Error setting up FFmpeg: {e}")

# --- NVIDIA CUDA Path Fix ---
# CTranslate2 needs access to cublas64_12.dll and cudnn64_9.dll etc.
# If installed via pip (nvidia-cublas-cu12, nvidia-cudnn-cu12), we need to add them to PATH.
def setup_nvidia_paths():
    paths_added = []
    
    try:
        import nvidia.cublas
        # Use __path__ for namespace packages (list of paths)
        if hasattr(nvidia.cublas, '__path__') and nvidia.cublas.__path__:
            cublas_dir = list(nvidia.cublas.__path__)[0]
            cublas_bin = os.path.join(cublas_dir, "bin")
            if os.path.exists(cublas_bin):
                os.environ["PATH"] = cublas_bin + os.pathsep + os.environ["PATH"]
                paths_added.append(cublas_bin)
    except ImportError:
        pass
    
    try:
        import nvidia.cudnn
        if hasattr(nvidia.cudnn, '__path__') and nvidia.cudnn.__path__:
            cudnn_dir = list(nvidia.cudnn.__path__)[0]
            cudnn_bin = os.path.join(cudnn_dir, "bin")
            if os.path.exists(cudnn_bin):
                os.environ["PATH"] = cudnn_bin + os.pathsep + os.environ["PATH"]
                paths_added.append(cudnn_bin)
    except ImportError:
        pass
    
    if paths_added:
        print(f"Added NVIDIA library paths: {paths_added}")
    else:
        print("NVIDIA python libraries (cublas/cudnn) not found. Relying on system PATH for CUDA.")

setup_nvidia_paths()
# ----------------------------

from core.audio import extract_audio
from core.transcriber import Transcriber
from core.gemini_transcriber import GeminiTranscriber
from core.translator import GeminiTranslator
from core.subtitle import generate_srt
import tempfile
from core.downloader import VideoDownloader

st.set_page_config(page_title="Video Subtitle Generator", page_icon="🎬", layout="wide")

st.title("🎬 视频双语字幕生成器 (English -> Chinese)")
st.markdown("上传英文视频，利用 GPU 提取音频，Faster-Whisper 转录，并使用 Gemini 进行翻译。")

from dotenv import load_dotenv
load_dotenv()

# Sidebar settings
st.sidebar.header("设置")
env_api_key = os.getenv("GEMINI_API_KEY", "")
api_key = st.sidebar.text_input("Gemini API Key", value=env_api_key, type="password", help="在此输入你的 Google Gemini API Key")

# Proxy settings
# Based on v2ray/clash default, usually 10808 or 7890
env_http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or "http://127.0.0.1:10808"
proxy_url = st.sidebar.text_input("HTTP代理 (可选)", value=env_http_proxy, help="如: http://127.0.0.1:10808。如果您在中国大陆，通常需要设置代理。")

if proxy_url:
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url

# Model Settings
gemini_model_name = st.sidebar.selectbox("Gemini 模型", ["gemini-3-flash-preview", "gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"], index=0)
model_size = st.sidebar.selectbox("Whisper 模型大小", ["base", "small", "medium", "large-v3"], index=2)

use_local_model = st.sidebar.checkbox("使用本地模型 (HuggingFace)", value=False)
local_model_path = ""
if use_local_model:
    default_path = r"D:\HuggingFace\hub\models--Systran--faster-whisper-medium"
    local_model_path = st.sidebar.text_input("本地模型路径", value=default_path)
    
    # Resolve snapshot path if needed
    if os.path.exists(local_model_path):
        if not os.path.exists(os.path.join(local_model_path, "model.bin")):
            snapshots_dir = os.path.join(local_model_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = os.listdir(snapshots_dir)
                if snapshots:
                    # Use the first snapshot found
                    full_snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                    st.sidebar.info(f"Using snapshot: {snapshots[0]}")
                    local_model_path = full_snapshot_path
    else:
        st.sidebar.warning("路径不存在，将尝试直接使用该路径。")



transcriber_engine = st.sidebar.selectbox(
    "转录引擎", 
    [
        "Faster-Whisper 时间戳 + Gemini 翻译 (推荐)",
        "Faster-Whisper (本地)", 
        "Gemini (云端，时间戳不精准)"
    ], 
    index=0
)

if "Gemini" in transcriber_engine and "Faster-Whisper" not in transcriber_engine:
    # Pure Gemini mode
    device_type = "cloud"
    st.sidebar.info("使用 Gemini 进行转录（注意：时间戳可能不精准）")
else:
    # Faster-Whisper modes (local or hybrid)
    device_type = st.sidebar.radio("运行设备", ["cuda", "cpu"], index=0)


# File uploader
# Source Selection
st.divider()
input_method = st.radio("选择视频来源", ["📂 上传本地文件", "🔗 输入视频链接"], horizontal=True)

video_path = None
video_name = "video.mp4"

if input_method == "📂 上传本地文件":
    uploaded_file = st.file_uploader("上传视频文件", type=["mp4", "mov", "mkv", "avi"])
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        video_name = uploaded_file.name
        tfile.close()
        
        # Clear download session state to avoid confusion
        if "downloaded_video_path" in st.session_state:
            st.session_state.downloaded_video_path = None

else: # Link Input
    if "downloaded_video_path" not in st.session_state:
        st.session_state.downloaded_video_path = None
    
    video_url = st.text_input("请输入视频 URL (支持 YouTube, Twitter/X, Bilibili 等)")
            
    if st.button("⬇️ 下载视频"):
        if not video_url:
            st.error("请输入有效的链接")
        else:
            downloader = VideoDownloader()
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            def progress_callback(d):
                if d['status'] == 'downloading':
                    try:
                        if d.get('total_bytes'):
                            p = d['downloaded_bytes'] / d['total_bytes']
                            progress_bar.progress(min(p, 1.0))
                    except:
                        pass
                    status_text.text(f"下载进度: {d.get('_percent_str', '0%')}")
                elif d['status'] == 'finished':
                    progress_bar.progress(1.0)
                    status_text.text("下载完成！")

            try:
                with st.spinner("正在下载视频..."):
                    current_proxy = proxy_url if proxy_url else None
                    
                    downloaded_path = downloader.download_video(
                        video_url, 
                        proxy=current_proxy,
                        progress_callback=progress_callback
                    )
                    st.session_state.downloaded_video_path = downloaded_path
                    st.success(f"✅ 已下载: {os.path.basename(downloaded_path)}")
                    st.info(f"📁 文件保存位置: {downloaded_path}")
                    
                    # Rerun to show video preview
                    st.rerun()
                        
            except Exception as e:
                st.error(f"下载失败: {e}")

    # Restore from session state if available
    if st.session_state.downloaded_video_path and os.path.exists(st.session_state.downloaded_video_path):
        video_path = st.session_state.downloaded_video_path
        video_name = os.path.basename(video_path)


if video_path is not None:
    st.video(video_path)

    if st.button("开始生成字幕"):
        if not api_key:
            st.error("请在左侧侧边栏输入 Gemini API Key！")
        else:
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()

                # 1. Extract Audio
                status_text.text("正在提取音频...")
                audio_path = extract_audio(video_path)
                st.success(f"音频提取成功: {audio_path}")
                progress_bar.progress(20)

                # 2. Transcribe
                segments = []
                if "Faster-Whisper" in transcriber_engine:
                    final_model_size = local_model_path if use_local_model and local_model_path else model_size
                    compute_type = "float16" if device_type == "cuda" else "int8"
                    status_text.text(f"正在转录音频 (模型: {final_model_size}, 设备: {device_type}, 精度: {compute_type})...")
                    transcriber = Transcriber(model_size=final_model_size, device=device_type, compute_type=compute_type)
                    
                    with st.spinner("Whisper 正在努力听写中..."):
                        segments = transcriber.transcribe(audio_path)
                else:
                    # Gemini Transcriber
                    status_text.text("正在使用 Gemini 进行转录和说话人分离...")
                    gemini_transcriber = GeminiTranscriber(api_key=api_key, model_name="gemini-3-flash-preview")
                    
                    with st.spinner("Gemini 正在聆聽并分析说话人..."):
                        segments = gemini_transcriber.transcribe(audio_path)
                
                st.success(f"转录完成，共 {len(segments)} 个片段。")
                progress_bar.progress(40)

                # 2.5 Refine Segments (optional, for Gemini mode only)
                # When using Faster-Whisper, timestamps are already accurate
                # When using Gemini, timestamps may need refinement
                if "Gemini" in transcriber_engine and "Faster-Whisper" not in transcriber_engine:
                    status_text.text("正在优化字幕分段...")
                    from core.subtitle import refine_segments
                    segments = refine_segments(segments, max_words=15, max_cjk_chars=30)
                    st.info(f"分段优化完成，共 {len(segments)} 个片段。")
                
                progress_bar.progress(50)

                # 3. Translate
                status_text.text(f"正在使用 {gemini_model_name} 进行翻译...")
                translator = GeminiTranslator(api_key=api_key, model_name=gemini_model_name)
                
                with st.spinner(f"{gemini_model_name} 正在翻译中..."):
                    translated_segments = translator.translate_segments(segments)
                
                st.success("翻译完成！")
                progress_bar.progress(80)

                # 4. Generate SRT
                status_text.text("正在生成字幕文件...")
                base_name = os.path.splitext(video_name)[0]
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                
                # English SRT
                srt_en_path = os.path.join(output_dir, f"{base_name}_en.srt")
                generate_srt(translated_segments, srt_en_path)
                
                # Chinese SRT
                # Construct segments for Zh only
                segments_zh = []
                for seg in translated_segments:
                    segments_zh.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg.get("text_zh", ""),
                        "speaker": seg.get("speaker", "")
                    })
                srt_zh_path = os.path.join(output_dir, f"{base_name}_zh.srt")
                generate_srt(segments_zh, srt_zh_path)

                # Bilingual SRT
                # Show English on top, Chinese on bottom usually, or combined line
                segments_bi = []
                for seg in translated_segments:
                    combined_text = f"{seg['text']}\n{seg.get('text_zh', '')}"
                    segments_bi.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": combined_text
                    })
                srt_bi_path = os.path.join(output_dir, f"{base_name}_bi.srt")
                generate_srt(segments_bi, srt_bi_path)

                progress_bar.progress(100)
                status_text.text("处理完成！")

                # Download Buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    with open(srt_en_path, "r", encoding="utf-8") as f:
                        st.download_button("下载英文字幕 (.srt)", f, file_name=f"{base_name}_en.srt")
                with col2:
                    with open(srt_zh_path, "r", encoding="utf-8") as f:
                        st.download_button("下载中文字幕 (.srt)", f, file_name=f"{base_name}_zh.srt")
                with col3:
                    with open(srt_bi_path, "r", encoding="utf-8") as f:
                        st.download_button("下载双语字幕 (.srt)", f, file_name=f"{base_name}_bi.srt")

                # 5. Video Synthesis
                st.markdown("---")
                st.subheader("🎬 视频预览与下载")
                status_text.text("正在合成视频字幕 (这可能需要一些时间)...")
                
                from core.video import burn_subtitles
                output_video_path = os.path.join(output_dir, f"{base_name}_subtitled.mp4")
                
                with st.spinner("正在将字幕烧录到视频中..."):
                    # Burn bilingual subtitles by default
                    burn_subtitles(video_path, srt_bi_path, output_video_path)
                
                st.success("视频合成完成！")
                st.video(output_video_path)
                
                with open(output_video_path, "rb") as f:
                    st.download_button(
                        label="下载带字幕的视频 (.mp4)",
                        data=f,
                        file_name=f"{base_name}_subtitled.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:
                st.error(f"发生错误: {str(e)}")
            finally:
                # Cleanup temp video file if desired, but user might want to see it
                pass
