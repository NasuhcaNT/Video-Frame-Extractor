import cv2
import os


def get_video_metadata(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Video açılamadı!")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": duration_sec,
        "resolution": (width, height),
    }


def extract_frames(
    video_path: str,
    output_dir: str,
    start_frame: int,
    end_frame: int,
    target_fps: float,
):
    cap = cv2.VideoCapture(video_path)

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = source_fps / target_fps

    os.makedirs(output_dir, exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    saved = 0
    current_frame = start_frame
    next_capture_frame = start_frame

    while cap.isOpened() and current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame >= next_capture_frame:
            filename = f"frame_{current_frame:06d}_{saved:06d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved += 1
            next_capture_frame += frame_interval

        current_frame += 1

    cap.release()

    print(f"✅ Kaydedilen kare sayısı: {saved}")


video_path = "video.mp4"
output_dir = "images"

# 1. Meta bilgileri al
meta = get_video_metadata(video_path)

print("🎥 Video Bilgileri")
for k, v in meta.items():
    print(f"{k}: {v}")

# 2. Frame çıkar
extract_frames(
    video_path=video_path,
    output_dir=output_dir,
    start_frame=300,
    end_frame=900,
    target_fps=2,  # saniyede 2 kare
)
