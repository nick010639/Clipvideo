import yt_dlp
import os

class VideoDownloader:
    def __init__(self, output_dir="downloads"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def download_video(self, url, proxy=None, cookies_from_browser=None, cookies_file_path=None, progress_callback=None):
        """
        Downloads a video from the given URL.
        
        Args:
            url (str): The video URL.
            proxy (str, optional): Proxy URL (e.g., http://127.0.0.1:10808).
            cookies_from_browser (str, optional): Browser to extract cookies from (e.g., 'chrome', 'firefox').
            cookies_file_path (str, optional): Path to a Netscape formatted cookies.txt file.
            progress_callback (function, optional): Callback function for progress updates.
                                                  It receives a dictionary with 'status' and other info.
        
        Returns:
            str: Path to the downloaded file.
        """
        
        # yt-dlp options
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(self.output_dir, '%(title)s.%(ext)s'),
            'noplaylist': True,
            # Network stability options
            'retries': 10,
            'fragment_retries': 10,
            'http_chunk_size': 10485760,  # 10MB
            'socket_timeout': 30,
        }
        
        if proxy:
            ydl_opts['proxy'] = proxy
            
        if cookies_from_browser:
            ydl_opts['cookiesfrombrowser'] = (cookies_from_browser,)
            
        if cookies_file_path:
            ydl_opts['cookiefile'] = cookies_file_path

        if progress_callback:
            def progress_hook(d):
                if d['status'] == 'downloading':
                   progress_callback(d)
                elif d['status'] == 'finished':
                   progress_callback(d)
            
            ydl_opts['progress_hooks'] = [progress_hook]

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                return filename
        except Exception as e:
            error_msg = str(e)
            # 检测Chrome cookie权限问题
            if "Could not copy Chrome cookie database" in error_msg or "Permission denied" in error_msg:
                raise Exception(
                    "Chrome Cookie 读取失败！这是因为 Chrome 浏览器正在运行，cookie 数据库被锁定。\n\n"
                    "解决方案（选择其一）：\n"
                    "1. 【推荐】将 Cookie来源浏览器 改为 'firefox'（Firefox 不会锁定数据库）\n"
                    "2. 关闭所有 Chrome 窗口后再尝试下载\n"
                    "3. 清空 Cookie来源浏览器 字段，不使用 cookies 下载（适用于公开视频）\n"
                    "4. 使用浏览器扩展导出 cookies.txt 文件，然后填写 Cookies文件路径"
                )
            raise Exception(f"下载失败: {error_msg}")
