from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_shared_recognition_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dicom-root", type=Path, required=True, help="Root containing category/subject/series DICOM folders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for manifests, previews and per-subject results.")
    parser.add_argument("--model-path", required=True, help="Local Hugging Face model directory, e.g. Lingshu 32B.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-series-per-subject", type=int, default=32)
    parser.add_argument("--max-subjects-per-category", type=int, default=400)
    parser.add_argument("--target-category")
    parser.add_argument("--target-subject")
    parser.add_argument("--no-resume", action="store_true", help="Reprocess subjects even when _COMPLETE exists.")
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=896)
    parser.add_argument("--runtime-reserve-gib", type=float, default=5.0)
    parser.add_argument("--cpu-offload-gib", type=int, default=96)
    parser.add_argument("--allow-model-download", action="store_true", help="Allow Transformers to resolve model files remotely.")


def _add_shared_conversion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest-csv", type=Path, required=True, help="recognize output manifests/dcm2nii_manifest.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="NIfTI output root.")
    parser.add_argument("--dcm2niix", default="dcm2niix", help="dcm2niix executable or absolute path.")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-unknown-plane", action="store_true")
    parser.add_argument("--non-strict-flair", action="store_true", help="Do not require FLAIR metadata validation.")
    parser.add_argument("--axial-cos-threshold", type=float, default=0.85)
    parser.add_argument("--min-ti-ms", type=float, default=1500.0)
    parser.add_argument("--min-te-ms", type=float, default=50.0)
    parser.add_argument("--min-tr-ms", type=float, default=4000.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mri-sequence", description="Clinical MRI sequence identification and DICOM-to-NIfTI conversion.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    recognize = subparsers.add_parser("recognize", help="Identify DICOM MRI sequences with a local VLM.")
    _add_shared_recognition_args(recognize)
    convert = subparsers.add_parser("convert", help="Select same-Study axial T1/T1CE/T2/FLAIR and convert with dcm2niix.")
    _add_shared_conversion_args(convert)
    pipeline = subparsers.add_parser("pipeline", help="Run recognize followed by convert.")
    _add_shared_recognition_args(pipeline)
    pipeline.add_argument("--nifti-output-dir", type=Path, required=True)
    pipeline.add_argument("--dcm2niix", default="dcm2niix")
    pipeline.add_argument("--timeout-seconds", type=int, default=600)
    pipeline.add_argument("--allow-unknown-plane", action="store_true")
    pipeline.add_argument("--non-strict-flair", action="store_true")
    pipeline.add_argument("--axial-cos-threshold", type=float, default=0.85)
    return parser


def _recognizer_config(args: argparse.Namespace):
    from .recognizer import RecognizerConfig
    return RecognizerConfig(
        dicom_root=args.dicom_root, output_dir=args.output_dir, model_path=args.model_path,
        image_size=args.image_size, max_series_per_subject=args.max_series_per_subject,
        max_subjects_per_category=None if args.max_subjects_per_category < 0 else args.max_subjects_per_category,
        target_category=args.target_category, target_subject=args.target_subject,
        resume=not args.no_resume, save_previews=not args.no_previews,
        runtime_reserve_gib=args.runtime_reserve_gib, cpu_offload_gib=args.cpu_offload_gib,
        max_new_tokens=args.max_new_tokens, local_files_only=not args.allow_model_download,
    )


def _conversion_config(args: argparse.Namespace, manifest: Path | None = None, output: Path | None = None):
    from .converter import ConversionConfig
    return ConversionConfig(
        manifest_csv=manifest or args.manifest_csv, output_dir=output or args.output_dir,
        dcm2niix=args.dcm2niix, timeout_seconds=args.timeout_seconds, resume=not args.no_resume,
        axial_cos_threshold=args.axial_cos_threshold, allow_unknown_plane=args.allow_unknown_plane,
        strict_flair_validation=not args.non_strict_flair,
        flair_min_ti_ms=getattr(args, "min_ti_ms", 1500.0), flair_min_te_ms=getattr(args, "min_te_ms", 50.0),
        flair_min_tr_ms=getattr(args, "min_tr_ms", 4000.0),
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "recognize":
        from .recognizer import run_recognition
        result = run_recognition(_recognizer_config(args))
    elif args.command == "convert":
        from .converter import run_conversion
        result = run_conversion(_conversion_config(args))
    else:
        from .converter import run_conversion
        from .recognizer import run_recognition
        recognition = run_recognition(_recognizer_config(args))
        manifest = Path(recognition["dcm2nii_manifest"])
        result = run_conversion(_conversion_config(args, manifest=manifest, output=args.nifti_output_dir))
    for name, path in result.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
