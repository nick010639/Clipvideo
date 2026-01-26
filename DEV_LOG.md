# Project Development Log

## 2026-01-25

### 1. 本地 Whisper 模型配置优化
- **目标**: 默认使用本地下载好的 Whisper Medium 模型，避免每次重复检查或下载。
- **改动**:
  - 在 `.env` 文件中添加了 `LOCAL_MODEL_PATH` 变量。
  - 修改 `app.py`，默认勾选 "使用本地模型"，并优先读取环境变量中的路径。
  - **经验**: 本地模型路径通常包含 `snapshots/<commit_hash>` 结构，代码中已包含自动解析这层结构的逻辑，确保能找到 `model.bin`。

### 2. 视频下载稳定性修复
- **问题**: 使用 `yt-dlp` 下载视频时，常遇到下载中断 (`IncompleteRead`) 或 "Downloaded X bytes, expected Y bytes" 错误，尤其是在网络波动时。
- **解决方案**: 在 `core/downloader.py` 中增强了 `yt_dlp` 的配置参数：
  ```python
  'retries': 10,              # 整体重试次数
  'fragment_retries': 10,     # 分片重试次数
  'http_chunk_size': 10485760,# 10MB 数据块，保持连接活跃
  'socket_timeout': 30,       # 30秒超时，避免过早断开
  ```
- **效果**: 显著提高了下载成功率，能够自动从中断处恢复或重试。

---

## 2026-01-26

### 1. FFmpeg 文件名特殊字符导致烧录字幕失败

- **问题**: 处理包含单引号 `'` 的文件名时（如 `This guy literally dropped the best life advice you'll ever hear.mp4`），FFmpeg 报错 `Unable to open ... .srt` 和 `Error opening output files: No such file or directory`。
  
- **根本原因**: 
  - FFmpeg 的 `subtitles` 滤镜使用单引号包裹路径参数，当文件名本身包含单引号时，会破坏命令语法。
  - 输出路径中的单引号也导致 FFmpeg 无法正确创建文件。

- **解决方案**:
  1. **`app.py`**: 在生成 `base_name` 后，使用正则表达式清理特殊字符：
     ```python
     import re
     base_name = re.sub(r"['\"]", "", base_name)  # 移除引号
     base_name = re.sub(r"[<>:\"|?*]", "_", base_name)  # 替换Windows非法字符
     ```
  2. **`core/video.py`**: 添加 `sanitize_filename()` 辅助函数，并确保输出目录存在。

- **经验教训**:
  - 处理用户输入的文件名时，始终要考虑特殊字符的影响。
  - FFmpeg 滤镜参数中的路径转义规则复杂，最简单的办法是在源头清理文件名。
  - Windows 路径中 `:` 需要转义为 `\:`，`\` 需要转为 `/`。
