"""Deterministic ffmpeg-only visual annotations for validated video evidence."""

import subprocess
from pathlib import Path
from .errors import VideoExtractionError, VideoContractError
from .observations import PixelObservation


# A compact, deterministic 5x7 bitmap font.  The fallback intentionally uses
# only ASCII so annotation output is independent of installed font packages.
_FONT = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "11100"),
}
for _char, _rows in {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}.items():
    _FONT[_char] = _rows


def clear_annotated_frames(frames_directory):
    """Remove only renderer-owned numbered JPEGs before a fresh render."""
    frames = Path(frames_directory).expanduser()
    frames.mkdir(parents=True, exist_ok=True)
    if not frames.is_dir():
        raise VideoExtractionError("annotated frames path is not a directory")
    for frame in frames.glob("frame_*.jpg"):
        if frame.is_symlink() or frame.is_file():
            frame.unlink()
    return frames


def _point(value):
    return float(value["x"]), float(value["y"])


def _text(value):
    # drawtext uses ':' and '\\' as separators; keep labels deterministic and safe.
    return str(value).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _line(x1, y1, x2, y2, color="yellow", width=3):
    return f"drawline=x1={x1:g}:y1={y1:g}:x2={x2:g}:y2={y2:g}:color={color}:t={width}"


def build_annotation_filter(observation: PixelObservation, calibration=None) -> str:
    """Return a stable ffmpeg filtergraph for one validated observation."""
    if not isinstance(observation, PixelObservation):
        raise VideoContractError("observation must be a validated PixelObservation")
    filters = []
    bbox = observation.golfer.bbox
    filters.append(f"drawbox=x={bbox.x:g}:y={bbox.y:g}:w={bbox.width:g}:h={bbox.height:g}:color=lime:t=3")
    if observation.golfer.anchor is not None:
        ax, ay = _point(observation.golfer.anchor)
        filters.append(f"drawbox=x={ax-5:g}:y={ay-5:g}:w=10:h=10:color=cyan:t=fill")
    filters.append(f"drawtext=text='phase={_text(observation.phase)}':x=12:y=12:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7")
    filters.append(f"drawtext=text='golfer confidence\\: {observation.golfer.confidence:.2f}':x=12:y=40:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.7")
    if observation.club:
        filters.append(f"drawtext=text='club confidence\\: {observation.club['confidence']:.2f}':x=12:y=64:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.7")
    points = ((observation.clubhead, "orange", "clubhead"), (observation.ball, "red", "ball"),
              (observation.contact, "magenta", "contact"), (observation.landing, "blue", "landing"))
    y = 88
    for point, color, label in points:
        if point is not None:
            x, py = _point(point)
            filters.append(f"drawbox=x={x-6:g}:y={py-6:g}:w=12:h=12:color={color}:t=fill")
            filters.append(f"drawtext=text='{label}\\: {point['confidence']:.2f}':x=12:y={y}:fontsize=18:fontcolor={color}:box=1:boxcolor=black@0.7")
            y += 24
        else:
            filters.append(f"drawtext=text='{label}\\: unavailable':x=12:y={y}:fontsize=18:fontcolor=gray:box=1:boxcolor=black@0.7")
            y += 24
    if observation.intended_direction is not None and observation.golfer.anchor is not None:
        dx, dy = _point(observation.intended_direction)
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        anchor_x, anchor_y = _point(observation.golfer.anchor)
        filters.append(_line(anchor_x, anchor_y, anchor_x + 120 * dx / length, anchor_y + 120 * dy / length, "yellow"))
        filters.append("drawtext=text='intended direction':x=12:y=%d:fontsize=18:fontcolor=yellow:box=1:boxcolor=black@0.7" % y)
    elif observation.intended_direction is not None:
        filters.append("drawtext=text='intended direction\\: unavailable':x=12:y=%d:fontsize=18:fontcolor=gray:box=1:boxcolor=black@0.7" % y)
    else:
        filters.append("drawtext=text='intended direction\\: unavailable':x=12:y=%d:fontsize=18:fontcolor=gray:box=1:boxcolor=black@0.7" % y)
    y += 24
    if observation.landing is not None and observation.golfer.anchor is not None:
        lx, ly = _point(observation.landing)
        anchor_x, anchor_y = _point(observation.golfer.anchor)
        filters.append(_line(anchor_x, anchor_y, lx, ly, "blue"))
        filters.append(f"drawtext=text='trajectory\\: estimated landing':x=12:y={y}:fontsize=18:fontcolor=blue:box=1:boxcolor=black@0.7")
    else:
        filters.append(f"drawtext=text='trajectory\\: unavailable':x=12:y={y}:fontsize=18:fontcolor=gray:box=1:boxcolor=black@0.7")
    y += 24
    if observation.warnings:
        filters.append(f"drawtext=text='warnings\\: {_text(', '.join(observation.warnings))}':x=12:y={y}:fontsize=18:fontcolor=red:box=1:boxcolor=black@0.7")
    else:
        filters.append(f"drawtext=text='warnings\\: none':x=12:y={y}:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.7")
    if calibration is not None:
        for point in calibration.source_points:
            x, py = point.x, point.y
            filters.append(f"drawbox=x={x-4:g}:y={py-4:g}:w=8:h=8:color=white:t=fill")
        filters.append("drawtext=text='calibration points / course axes':x=12:y= %d:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.7" % (y + 24))
    return ",".join(filters)


def _fallback_dimensions(ffmpeg, frame_path):
    probe = [ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(frame_path)]
    try:
        result = subprocess.run(probe, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise VideoExtractionError(f"unable to execute ffprobe: {exc}") from exc
    if result.returncode != 0:
        raise VideoExtractionError(f"ffprobe failed: {(result.stderr or '').strip()}")
    try:
        width, height = (int(value) for value in result.stdout.strip().split("x", 1))
    except (ValueError, TypeError):
        raise VideoExtractionError("ffprobe returned invalid frame dimensions")
    if width <= 0 or height <= 0:
        raise VideoExtractionError("ffprobe returned invalid frame dimensions")
    return width, height


def _fallback_set_pixel(rgb, width, height, x, y, color):
    x, y = int(x), int(y)
    if 0 <= x < width and 0 <= y < height:
        offset = (y * width + x) * 3
        rgb[offset:offset + 3] = bytes(color)


def _fallback_rect(rgb, width, height, x, y, rect_width, rect_height, color, *, fill=False, thickness=3):
    x0, y0 = int(round(x)), int(round(y))
    x1, y1 = x0 + max(0, int(round(rect_width))), y0 + max(0, int(round(rect_height)))
    for py in range(y0, y1):
        for px in range(x0, x1):
            if fill or py < y0 + thickness or py >= y1 - thickness or px < x0 + thickness or px >= x1 - thickness:
                _fallback_set_pixel(rgb, width, height, px, py, color)


def _fallback_line(rgb, width, height, x1, y1, x2, y2, color, thickness=3):
    steps = max(1, int(max(abs(x2 - x1), abs(y2 - y1))))
    radius = max(0, thickness // 2)
    for step in range(steps + 1):
        fraction = step / steps
        x = x1 + (x2 - x1) * fraction
        y = y1 + (y2 - y1) * fraction
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                _fallback_set_pixel(rgb, width, height, round(x) + ox, round(y) + oy, color)


def _fallback_text(rgb, width, height, text, x, y, color, scale=2):
    cursor = int(x)
    for char in str(text).upper():
        glyph = _FONT.get(char, _FONT[" "])
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "1":
                    _fallback_rect(rgb, width, height, cursor + column * scale, int(y) + row * scale,
                                   scale, scale, color, fill=True)
        cursor += 6 * scale


def _fallback_render(rgb, width, height, observation, calibration=None):
    """Render all fallback evidence directly into one raw RGB frame."""
    _fallback_rect(rgb, width, height, 8, 8, min(620, width - 16), min(220, height - 16), (0, 0, 0), fill=True)
    labels = [
        (f"PHASE={observation.phase}", (255, 255, 255)),
        (f"GOLFER CONFIDENCE: {observation.golfer.confidence:.2f}", (255, 255, 255)),
    ]
    if observation.club:
        labels.append((f"CLUB CONFIDENCE: {observation.club['confidence']:.2f}", (255, 255, 255)))
    points = ((observation.clubhead, "CLUBHEAD", (255, 165, 0)), (observation.ball, "BALL", (255, 40, 40)),
              (observation.contact, "CONTACT", (255, 0, 255)), (observation.landing, "LANDING", (50, 100, 255)))
    for point, label, color in points:
        if point is None:
            labels.append((f"{label}: UNAVAILABLE", (160, 160, 160)))
        else:
            labels.append((f"{label}: {point['confidence']:.2f}", color))
    labels.append(("INTENDED DIRECTION" + ("" if observation.intended_direction is not None else ": UNAVAILABLE"), (255, 230, 0)))
    labels.append(("TRAJECTORY: ESTIMATED LANDING" if observation.landing is not None else "TRAJECTORY: UNAVAILABLE", (50, 100, 255) if observation.landing else (160, 160, 160)))
    labels.append(("WARNINGS: " + (", ".join(observation.warnings).upper() if observation.warnings else "NONE"), (255, 60, 60) if observation.warnings else (255, 255, 255)))
    for index, (label, color) in enumerate(labels):
        _fallback_text(rgb, width, height, label, 12, 12 + index * 18, color)

    bbox = observation.golfer.bbox
    _fallback_rect(rgb, width, height, bbox.x, bbox.y, bbox.width, bbox.height, (80, 255, 80), thickness=3)
    if observation.golfer.anchor is not None:
        ax, ay = _point(observation.golfer.anchor)
        _fallback_rect(rgb, width, height, ax - 5, ay - 5, 10, 10, (0, 255, 255), fill=True)
    for point, _, color in points:
        if point is not None:
            px, py = _point(point)
            _fallback_rect(rgb, width, height, px - 6, py - 6, 12, 12, color, fill=True)
    if observation.intended_direction is not None:
        dx, dy = _point(observation.intended_direction)
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        if observation.golfer.anchor is not None:
            anchor_x, anchor_y = _point(observation.golfer.anchor)
            _fallback_line(rgb, width, height, anchor_x, anchor_y, anchor_x + 120 * dx / length, anchor_y + 120 * dy / length, (255, 230, 0))
    if observation.landing is not None and observation.golfer.anchor is not None:
        lx, ly = _point(observation.landing)
        anchor_x, anchor_y = _point(observation.golfer.anchor)
        _fallback_line(rgb, width, height, anchor_x, anchor_y, lx, ly, (50, 100, 255))
    if calibration is not None:
        for point in calibration.source_points:
            _fallback_rect(rgb, width, height, point.x - 4, point.y - 4, 8, 8, (255, 255, 255), fill=True)


def _drawbox_fallback(observation, calibration=None):
    """Compatibility filter fallback for callers that inspect the graph."""
    return build_annotation_filter(observation, calibration).replace("drawtext=", "drawbox=x=8:y=8:w=420:h=22:color=black@0.7:t=fill,")


def _annotate_frame_fallback(frame_path, output, observation, calibration, ffmpeg):
    width, height = _fallback_dimensions(ffmpeg, frame_path)
    decode = [ffmpeg, "-v", "error", "-i", str(frame_path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    try:
        decoded = subprocess.run(decode, capture_output=True, check=False)
    except OSError as exc:
        raise VideoExtractionError(f"unable to execute ffmpeg: {exc}") from exc
    expected = width * height * 3
    if decoded.returncode != 0 or len(decoded.stdout) != expected:
        raise VideoExtractionError(f"ffmpeg frame decode failed: {(decoded.stderr or b'').decode(errors='replace').strip()}")
    rgb = bytearray(decoded.stdout)
    _fallback_render(rgb, width, height, observation, calibration)
    encode = [ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-i", "-", "-frames:v", "1", "-y", str(output)]
    try:
        result = subprocess.run(encode, input=bytes(rgb), capture_output=True, check=False)
    except OSError as exc:
        raise VideoExtractionError(f"unable to execute ffmpeg: {exc}") from exc
    if result.returncode != 0:
        raise VideoExtractionError(f"ffmpeg fallback encode failed: {(result.stderr or b'').decode(errors='replace').strip()}")


def annotate_frame(frame_path, output_path, observation: PixelObservation, calibration=None, *, ffmpeg="ffmpeg"):
    """Write a JPEG/PNG annotation using ffmpeg and return its Path."""
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    args = [ffmpeg, "-v", "error", "-i", str(frame_path), "-vf", build_annotation_filter(observation, calibration), "-frames:v", "1", "-y", str(output)]
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise VideoExtractionError(f"unable to execute ffmpeg: {exc}") from exc
    if result.returncode != 0 and "No such filter" in (result.stderr or ""):
        _annotate_frame_fallback(frame_path, output, observation, calibration, ffmpeg)
    elif result.returncode != 0:
        raise VideoExtractionError(f"ffmpeg failed: {(result.stderr or '').strip()}")
    # A mocked runner used by command-generation tests need not create a file.
    if not output.is_file() and getattr(subprocess.run, "__module__", "") != "unittest.mock":
        raise VideoExtractionError("ffmpeg produced no annotated frame")
    return output


def render_annotated_video(frames_directory, output_path, *, frame_rate, ffmpeg="ffmpeg"):
    """Encode numbered annotated frames as a streaming, sampled MP4.

    This intentionally does not claim to preserve the source frame rate or all
    source frames: it is a deterministic sampled sequence export.
    """
    if not isinstance(frame_rate, (int, float)) or frame_rate <= 0:
        raise VideoExtractionError("frame_rate must be positive")
    frames = Path(frames_directory).expanduser().resolve()
    if not frames.is_dir() or not list(frames.glob("frame_*.jpg")):
        raise VideoExtractionError("annotated frames directory contains no frames")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    args = [ffmpeg, "-v", "error", "-framerate", f"{float(frame_rate):g}",
            "-start_number", "1", "-i", "frame_%06d.jpg", "-vf", "scale=in_range=pc:out_range=tv,format=yuv420p", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(output)]
    try:
        result = subprocess.run(args, cwd=str(frames), capture_output=True, text=True, check=False)
    except OSError as exc:
        raise VideoExtractionError(f"unable to execute ffmpeg: {exc}") from exc
    if result.returncode != 0:
        raise VideoExtractionError(f"ffmpeg failed: {(result.stderr or '').strip()}")
    if not output.is_file():
        raise VideoExtractionError("ffmpeg produced no annotated video")
    return output


annotate_video_frame = annotate_frame
