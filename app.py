import os
import re
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from core.audio import extract_audio
from core.downloader import VideoDownloader
from core.subtitle import generate_srt
from core.transcriber import Transcriber
from core.translator import build_translator


try:
    import imageio_ffmpeg

    ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_cmd)
    ffmpeg_shim = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(ffmpeg_shim):
        shutil.copy2(ffmpeg_cmd, ffmpeg_shim)
    os.environ["PATH"] += os.pathsep + ffmpeg_dir
except ImportError:
    pass
except Exception as e:
    print(f"Error setting up FFmpeg: {e}")


def setup_nvidia_paths():
    try:
        import nvidia.cublas

        if hasattr(nvidia.cublas, "__path__") and nvidia.cublas.__path__:
            cublas_dir = list(nvidia.cublas.__path__)[0]
            cublas_bin = os.path.join(cublas_dir, "bin")
            if os.path.exists(cublas_bin):
                os.environ["PATH"] = cublas_bin + os.pathsep + os.environ["PATH"]
    except ImportError:
        pass

    try:
        import nvidia.cudnn

        if hasattr(nvidia.cudnn, "__path__") and nvidia.cudnn.__path__:
            cudnn_dir = list(nvidia.cudnn.__path__)[0]
            cudnn_bin = os.path.join(cudnn_dir, "bin")
            if os.path.exists(cudnn_bin):
                os.environ["PATH"] = cudnn_bin + os.pathsep + os.environ["PATH"]
    except ImportError:
        pass


setup_nvidia_paths()
load_dotenv()

st.set_page_config(page_title="Video Subtitle Generator", page_icon="🎬", layout="wide")

st.title("🎬 视频双语字幕生成器")
st.markdown("使用 Faster-Whisper 本地转写，并按需选择 Ollama 或 MiniMax 来做字幕翻译。")

st.sidebar.header("设置")

translation_provider = st.sidebar.radio(
    "翻译后端",
    ["Ollama (Local)", "MiniMax (Cloud)"],
    index=0,
)
translation_provider_key = "ollama" if translation_provider.startswith("Ollama") else "minimax"

env_http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or "http://127.0.0.1:10808"
proxy_url = st.sidebar.text_input(
    "HTTP代理 (可选)",
    value=env_http_proxy,
    help="如: http://127.0.0.1:10808。如果你在中国大陆且使用云端翻译，通常需要设置代理。",
)
if proxy_url:
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url

if translation_provider_key == "ollama":
    ollama_base_url = st.sidebar.text_input(
        "Ollama 地址",
        value=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    translation_model_name = st.sidebar.text_input(
        "Ollama 模型",
        value=os.getenv("OLLAMA_MODEL", "qwen3.5:9b"),
    )
    api_key = ""
    translator_base_url = ollama_base_url
    st.sidebar.info("默认使用本地 Ollama 翻译，适合长视频，避免云端额度和速率限制。")
else:
    env_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    api_key = st.sidebar.text_input(
        "MiniMax API Key",
        value=env_api_key,
        type="password",
        help="在此输入你的 MiniMax Token Plan API Key",
    )
    translation_model_name = st.sidebar.text_input(
        "MiniMax 模型",
        value=os.getenv("MINIMAX_MODEL", "MiniMax-M2.7"),
    )
    translator_base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic")
    st.sidebar.info("MiniMax 作为备用云端翻译选项保留。")

st.sidebar.markdown("---")
model_size = st.sidebar.selectbox("Whisper 模型大小", ["base", "small", "medium", "large-v3"], index=2)
use_local_model = st.sidebar.checkbox("使用本地模型 (HuggingFace)", value=True)

local_model_path = ""
if use_local_model:
    default_path = os.getenv("LOCAL_MODEL_PATH", r"D:\HuggingFace\hub\models--Systran--faster-whisper-medium")
    local_model_path = st.sidebar.text_input("本地模型路径", value=default_path)
    if os.path.exists(local_model_path) and not os.path.exists(os.path.join(local_model_path, "model.bin")):
        snapshots_dir = os.path.join(local_model_path, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                local_model_path = os.path.join(snapshots_dir, snapshots[0])
                st.sidebar.info(f"Using snapshot: {snapshots[0]}")
    elif local_model_path and not os.path.exists(local_model_path):
        st.sidebar.warning("本地模型路径不存在，将尝试直接使用该路径。")

device_type = st.sidebar.radio("运行设备", ["cuda", "cpu"], index=0)

st.divider()
input_method = st.radio("选择视频来源", ["📁 上传本地文件", "🔗 输入视频链接"], horizontal=True)

video_path = None
video_name = "video.mp4"

if input_method == "📁 上传本地文件":
    uploaded_file = st.file_uploader("上传视频文件", type=["mp4", "mov", "mkv", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        video_name = uploaded_file.name
        tfile.close()
        if "downloaded_video_path" in st.session_state:
            st.session_state.downloaded_video_path = None
else:
    if "downloaded_video_path" not in st.session_state:
        st.session_state.downloaded_video_path = None

    video_url = st.text_input("请输入视频 URL (支持 YouTube, Twitter/X, Bilibili 等)")
    if st.button("⬇️ 下载视频"):
        if not video_url:
            st.error("请输入有效的视频链接")
        else:
            downloader = VideoDownloader()
            status_text = st.empty()
            progress_bar = st.progress(0)

            def progress_callback(d):
                if d["status"] == "downloading":
                    try:
                        if d.get("total_bytes"):
                            p = d["downloaded_bytes"] / d["total_bytes"]
                            progress_bar.progress(min(p, 1.0))
                    except Exception:
                        pass
                    status_text.text(f"下载进度: {d.get('_percent_str', '0%')}")
                elif d["status"] == "finished":
                    progress_bar.progress(1.0)
                    status_text.text("下载完成")

            try:
                with st.spinner("正在下载视频..."):
                    downloaded_path = downloader.download_video(
                        video_url,
                        proxy=proxy_url if proxy_url else None,
                        progress_callback=progress_callback,
                    )
                    st.session_state.downloaded_video_path = downloaded_path
                    st.success(f"已下载: {os.path.basename(downloaded_path)}")
                    st.info(f"文件保存位置: {downloaded_path}")
                    st.rerun()
            except Exception as e:
                st.error(f"下载失败: {e}")

    if st.session_state.downloaded_video_path and os.path.exists(st.session_state.downloaded_video_path):
        video_path = st.session_state.downloaded_video_path
        video_name = os.path.basename(video_path)

if video_path is not None:
    st.video(video_path)

    if st.button("开始生成字幕"):
        if translation_provider_key == "minimax" and not api_key:
            st.error("请在左侧输入 MiniMax API Key")
        else:
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("正在提取音频...")
                audio_path = extract_audio(video_path)
                st.success(f"音频提取成功: {audio_path}")
                progress_bar.progress(20)

                final_model_size = local_model_path if use_local_model and local_model_path else model_size
                compute_type = "float16" if device_type == "cuda" else "int8"
                status_text.text(
                    f"正在转录音频 (模型: {final_model_size}, 设备: {device_type}, 精度: {compute_type})..."
                )
                transcriber = Transcriber(
                    model_size=final_model_size,
                    device=device_type,
                    compute_type=compute_type,
                )

                with st.spinner("Whisper 正在转写中..."):
                    segments = transcriber.transcribe(audio_path)

                st.success(f"转录完成，共 {len(segments)} 个片段。")
                progress_bar.progress(50)

                translator = build_translator(
                    provider=translation_provider_key,
                    model_name=translation_model_name,
                    api_key=api_key,
                    base_url=translator_base_url,
                )

                status_text.text(f"正在使用 {translation_provider} / {translation_model_name} 翻译字幕...")
                with st.spinner(f"{translation_model_name} 正在翻译中..."):
                    translated_segments = translator.translate_segments(segments)

                st.success("翻译完成。")
                progress_bar.progress(80)

                status_text.text("正在生成字幕文件...")
                base_name = os.path.splitext(video_name)[0]
                base_name = re.sub(r"['\"]", "", base_name)
                base_name = re.sub(r'[<>:"|?*]', "_", base_name)
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)

                srt_en_path = os.path.join(output_dir, f"{base_name}_en.srt")
                generate_srt(translated_segments, srt_en_path)

                segments_zh = []
                for seg in translated_segments:
                    segments_zh.append(
                        {
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg.get("text_zh", ""),
                            "speaker": seg.get("speaker", ""),
                        }
                    )
                srt_zh_path = os.path.join(output_dir, f"{base_name}_zh.srt")
                generate_srt(segments_zh, srt_zh_path)

                segments_bi = []
                for seg in translated_segments:
                    combined_text = f"{seg['text']}\n{seg.get('text_zh', '')}"
                    segments_bi.append(
                        {
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": combined_text,
                        }
                    )
                srt_bi_path = os.path.join(output_dir, f"{base_name}_bi.srt")
                generate_srt(segments_bi, srt_bi_path)

                progress_bar.progress(100)
                status_text.text("处理完成。")

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

                st.markdown("---")
                st.subheader("🎬 视频预览与下载")
                status_text.text("正在合成带字幕视频，这可能需要一点时间...")

                from core.video import burn_subtitles

                output_video_path = os.path.join(output_dir, f"{base_name}_subtitled.mp4")
                with st.spinner("正在将字幕烧录到视频中..."):
                    burn_subtitles(video_path, srt_bi_path, output_video_path)

                st.success("视频合成完成。")
                st.video(output_video_path)

                with open(output_video_path, "rb") as f:
                    st.download_button(
                        label="下载带字幕的视频 (.mp4)",
                        data=f,
                        file_name=f"{base_name}_subtitled.mp4",
                        mime="video/mp4",
                    )
            except Exception as e:
                st.error(f"发生错误: {str(e)}")
                print(f"App Error: {e}")
