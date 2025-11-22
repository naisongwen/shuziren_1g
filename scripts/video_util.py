import cv2
from moviepy.editor import VideoFileClip

def get_video_info(video_path):
    # 获取宽高和帧率
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 获取时长
    clip = VideoFileClip(video_path)
    duration = clip.duration
    clip.close()

    return width, height, fps, duration,frame_count