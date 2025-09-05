import os
import subprocess
from typing import Union


def extract_audio_from_mp4(
        video_path:str, 
        output_audio_path:Union[str, None]=None, 
        audio_format:str="mp3"
        ) -> None:
    """
    从MP4文件提取音频
    
    Parameters:
    video_path (str): Path to the MP4 video file.
    output_audio_path (str): Path for the output audio file. If None, it will save in same folder as video.
    audio_format (str): Desired audio format ('mp3', 'wav', 'aac', etc.)

    Returns:
    str: Path to the saved audio file.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")

    # Default output file name
    if output_audio_path is None:
        base, _ = os.path.splitext(video_path)
        output_audio_path = f"{base}.{audio_format}"

    # 调用 ffmpeg
    command = [
        "ffmpeg",
        "-i", video_path,   # input file
        "-vn",              # disable video
        "-acodec", "copy" if audio_format in ["aac", "m4a"] else "libmp3lame",
        output_audio_path
    ]

    subprocess.run(command, check=True)

def ms_to_time_str(ms:float):
    """
    将毫秒数转换为时间字符串
    """
    import datetime
    seconds:float = ms / 1000.0
    return str(datetime.timedelta(seconds=seconds)).split('.')[0]

def rename_video(video_path: str):
    '''
    从B站视频文件名，提取视频标题
    '''
    mp3_files = os.listdir(video_path)
    return [mp3_file.split(' - ')[1].split('(Av')[0] for mp3_file in mp3_files]

if __name__ == "__main__":
    extract_audio_from_mp4("/mnt/e/LLM/video_data/南娘大赛3.0质量越来越顶了,主播依旧完败 [BV1r8bNzuEeL].mp4", audio_format="mp3")
# extract_audio_from_mp4("谁是造成年轻人电子阳痿的元凶？【丰言疯话】 [BV1Nhe9zhEmb].mp4", audio_format="wav")
# print(f"Audio saved at: {audio_file}")