import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.robustness.codecs import available_conditions, generate_robustness_dataset


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate independent MP3/AAC/Opus, resampling, additive-noise, "
            "and telephony variants from an audio manifest."
        )
    )
    parser.add_argument("--manifest", help="Input CSV with name, label, and split columns")
    parser.add_argument("--output-root", help="Directory for generated audio and manifest")
    parser.add_argument(
        "--audio-root",
        default=None,
        help="Optional root directory used to resolve input audio paths",
    )
    parser.add_argument(
        "--conditions",
        default="all",
        help="Comma-separated condition names, or 'all'. See --list-conditions.",
    )
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="List available conditions and exit.",
    )
    parser.add_argument("--target-sample-rate", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ffmpeg", default=None, help="Optional path to the FFmpeg executable")
    parser.add_argument(
        "--omit-clean",
        action="store_true",
        help="Do not include clean source rows in the output manifest",
    )
    args = parser.parse_args()

    if args.list_conditions:
        print("\n".join(available_conditions()))
        return
    if not args.manifest or not args.output_root:
        parser.error("--manifest and --output-root are required unless --list-conditions is used")

    conditions = None
    if args.conditions.lower() != "all":
        conditions = [name.strip() for name in args.conditions.split(",") if name.strip()]

    result = generate_robustness_dataset(
        manifest_path=args.manifest,
        output_root=args.output_root,
        audio_root=args.audio_root,
        conditions=conditions,
        target_sample_rate=args.target_sample_rate,
        seed=args.seed,
        include_clean=not args.omit_clean,
        ffmpeg_path=args.ffmpeg,
    )
    print(f"Generated {len(result)} manifest rows.")
    print(f"Manifest: {Path(args.output_root).resolve() / 'robustness_manifest.csv'}")


if __name__ == "__main__":
    main()
