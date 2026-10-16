
from pathlib import Path
import shutil
import subprocess


CODEC_SETTINGS = {
    "mp3": {
        "64": ["-codec:a", "libmp3lame", "-b:a", "64k"],
        "128": ["-codec:a", "libmp3lame", "-b:a", "128k"],
        "192": ["-codec:a", "libmp3lame", "-b:a", "192k"],
    },
    "aac": {
        "32": ["-codec:a", "aac", "-b:a", "32k"],
        "48": ["-codec:a", "aac", "-b:a", "48k"],
        "64": ["-codec:a", "aac", "-b:a", "64k"],
    },
    "opus": {
        "32": ["-codec:a", "libopus", "-b:a", "32k"],
        "64": ["-codec:a", "libopus", "-b:a", "64k"],
        "128": ["-codec:a", "libopus", "-b:a", "128k"],
    },
}


def compress_audio(
    input_path,
    output_path,
    codec,
    bitrate,
    ffmpeg_path=None,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    codec = str(codec).lower()
    bitrate = str(bitrate)

    if codec not in CODEC_SETTINGS:
        raise ValueError(
            f"Unsupported codec: {codec}. "
            f"Supported codecs: {list(CODEC_SETTINGS)}"
        )

    if bitrate not in CODEC_SETTINGS[codec]:
        raise ValueError(
            f"Unsupported bitrate {bitrate} for {codec}. "
            f"Supported bitrates: {list(CODEC_SETTINGS[codec])}"
        )

    if not input_path.is_file():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    if ffmpeg_path is not None:
        ffmpeg = str(ffmpeg_path)
    else:
        ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg or not Path(ffmpeg).is_file():
        raise FileNotFoundError(
            "FFmpeg executable not found. Set ffmpeg_path "
            "or add FFmpeg to PATH."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(input_path),
        "-map", "0:a:0",
        "-vn",
        *CODEC_SETTINGS[codec][bitrate],
    ]

    if codec == "mp3":
        command += ["-f", "mp3"]
    elif codec == "aac":
        command += ["-f", "ipod"]
    elif codec == "opus":
        command += ["-f", "ogg"]

    command.append(str(output_path))

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"FFmpeg failed for {codec} at {bitrate} kbps.\n"
            f"Command: {' '.join(command)}\n"
            f"Error: {result.stderr}"
        )

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"FFmpeg did not produce a valid output: {output_path}"
        )

    return output_path