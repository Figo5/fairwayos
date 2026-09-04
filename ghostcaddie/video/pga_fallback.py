"""Portable, pixel-visible fallback rendering for the research-only demo."""
from pathlib import Path


def render_pga_fallback(source, destination, *, max_frames=None, pose_by_frame=None):
    """Copy local frames through OpenCV with truthful research annotations."""
    import cv2

    source, destination = Path(source), Path(destination)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        # Preserve the failure package even when a test/provider hands us a bad file.
        cap.release()
        width, height, fps = 320, 240, 1.0
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        frame = __import__('numpy').zeros((height, width, 3), dtype='uint8')
        cv2.putText(frame, "SOURCE UNAVAILABLE", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        writer.release()
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise OSError("video has no usable dimensions")
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise OSError("OpenCV cannot create MP4 writer")
    index = 0
    try:
        while max_frames is None or index < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = index / fps
            pose = (pose_by_frame or {}).get(index)
            cv2.rectangle(frame, (8, 8), (min(width - 8, 390), min(height - 8, 125)), (0, 0, 0), -1)
            lines = ["PGA RESEARCH DEMO", "NOT VALIDATED | RESEARCH ONLY",
                     "NO PRODUCTION ANALYTICS", f"FRAME {index:06d}  T+{timestamp:07.3f}s"]
            if pose:
                points = [tuple(map(int, p)) for p in pose]
                for a, b in zip(points, points[1:]):
                    cv2.line(frame, a, b, (255, 180, 0), 2)
                x = min(p[0] for p in points)
                y = min(p[1] for p in points)
                w = max(p[0] for p in points) - x
                h = max(p[1] for p in points) - y
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.circle(frame, points[0], 5, (0, 255, 255), -1)
                lines.append("POSE: SKELETON / BOX / ANCHOR")
            else:
                lines.append("POSE: UNAVAILABLE")
            lines += ["BALL: UNAVAILABLE", "CLUBHEAD: UNAVAILABLE"]
            for row, text in enumerate(lines):
                cv2.putText(frame, text, (15, 25 + row * 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            index += 1
    finally:
        cap.release()
        writer.release()
    if index == 0:
        raise OSError("video contains no decodable frames")
    return index
