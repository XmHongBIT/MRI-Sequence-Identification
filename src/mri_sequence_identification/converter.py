from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pydicom
from tqdm import tqdm

from .dicom import read_header
from .utils import atomic_csv, atomic_json, bool_value, natural_key, number, safe_name, text


MODALITY_TO_FILENAME = {"T1": "t1", "T1CE": "t1ce", "T2": "t2", "FLAIR": "t2flair"}
REQUIRED_MODALITIES = tuple(MODALITY_TO_FILENAME)


@dataclass
class ConversionConfig:
    manifest_csv: Path
    output_dir: Path
    dcm2niix: str = "dcm2niix"
    workers: int = 1
    timeout_seconds: int = 600
    resume: bool = True
    require_same_study: bool = True
    axial_cos_threshold: float = 0.85
    allow_unknown_plane: bool = False
    strict_flair_validation: bool = True
    flair_min_ti_ms: float = 1500.0
    flair_min_te_ms: float = 50.0
    flair_min_tr_ms: float = 4000.0
    min_nifti_file_size_bytes: int = 10 * 1024
    min_valid_slices: int = 5
    max_valid_voxel_spacing_mm: float = 10.0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _one_dicom(row: dict[str, str]) -> str:
    candidate = text(row.get("middle_dicom_abs"))
    if candidate and Path(candidate).is_file():
        return candidate
    filelist = text(row.get("filelist_abs"))
    if filelist and Path(filelist).is_file():
        for line in Path(filelist).read_text(encoding="utf-8").splitlines():
            if Path(line.strip()).is_file():
                return line.strip()
    folder = text(row.get("source_folder_abs"))
    if folder:
        for path in Path(folder).iterdir():
            if path.is_file():
                try:
                    read_header(path)
                    return str(path)
                except Exception:
                    continue
    return ""


def _refresh_metadata(row: dict[str, str]) -> dict[str, str]:
    path = _one_dicom(row)
    if not path:
        return row
    try:
        ds = read_header(path)
    except Exception:
        return row
    for key in ("StudyInstanceUID", "SeriesInstanceUID", "SeriesDescription", "ProtocolName", "ImageType", "RepetitionTime", "EchoTime", "InversionTime", "ContrastBolusAgent", "PixelSpacing", "SliceThickness", "ImageOrientationPatient", "ImagePositionPatient"):
        value = getattr(ds, key, "")
        if isinstance(value, (list, tuple)):
            value = ",".join(text(item) for item in value)
        if value not in (None, ""):
            row[key] = text(value)
    row["_middle_dicom_abs"] = path
    row["_plane"] = _plane(ds)
    return row


def _plane(ds: Any) -> str:
    try:
        orientation = np.asarray(getattr(ds, "ImageOrientationPatient"), dtype=float)
        if orientation.size != 6:
            return "UNKNOWN"
        normal = np.cross(orientation[:3], orientation[3:])
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            return "UNKNOWN"
        axis = int(np.argmax(np.abs(normal / norm)))
        return {0: "SAGITTAL", 1: "CORONAL", 2: "AXIAL"}[axis]
    except Exception:
        return "UNKNOWN"


def _is_axial(row: dict[str, str], threshold: float, allow_unknown: bool) -> bool:
    path = text(row.get("_middle_dicom_abs"))
    if not path:
        return allow_unknown
    try:
        ds = read_header(path)
        orientation = np.asarray(getattr(ds, "ImageOrientationPatient"), dtype=float)
        if orientation.size != 6:
            return allow_unknown
        normal = np.cross(orientation[:3], orientation[3:])
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            return allow_unknown
        return abs(float(normal[2] / norm)) >= threshold
    except Exception:
        return allow_unknown


def _is_derived(row: dict[str, str]) -> bool:
    image_type = text(row.get("ImageType")).upper()
    return any(token in image_type for token in ("DERIVED", "SECONDARY", "MPR", "MIP"))


def _flair_valid(row: dict[str, str], config: ConversionConfig) -> tuple[bool, str]:
    if not config.strict_flair_validation:
        return True, "strict_flair_validation_disabled"
    description = " ".join(text(row.get(key)) for key in ("SeriesDescription", "ProtocolName", "SequenceName", "ScanOptions")).upper()
    keyword = any(token in description for token in ("FLAIR", "TIRM", "IR-FLAIR", "FLUID ATTENUATED"))
    ti, te, tr = number(row.get("InversionTime")), number(row.get("EchoTime")), number(row.get("RepetitionTime"))
    physical = ti is not None and te is not None and tr is not None and ti >= config.flair_min_ti_ms and te >= config.flair_min_te_ms and tr >= config.flair_min_tr_ms
    if keyword:
        return True, "flair_keyword"
    if physical:
        return True, "flair_physical_parameters"
    return False, "missing_flair_keyword_or_physical_parameters"


def _effective_label(row: dict[str, str]) -> str:
    manual = text(row.get("manual_label")).upper()
    return manual if manual else text(row.get("final_label") or row.get("model_prediction")).upper()


def _candidate_score(row: dict[str, str], modality: str, config: ConversionConfig) -> tuple[int, int, int, int]:
    label = _effective_label(row)
    confidence = text(row.get("model_confidence")).upper()
    return (
        int(label == modality),
        int(confidence == "HIGH") * 2 + int(confidence == "MEDIUM"),
        int(not _is_derived(row)),
        int(number(row.get("n_dicoms"), 0) or 0),
    )


def select_four_modalities(rows: list[dict[str, str]], config: ConversionConfig) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    refreshed = [_refresh_metadata(dict(row)) for row in rows]
    candidates: list[dict[str, str]] = []
    for row in refreshed:
        label = _effective_label(row)
        if label not in REQUIRED_MODALITIES:
            continue
        if not _is_axial(row, config.axial_cos_threshold, config.allow_unknown_plane):
            row["_selection_exclusion"] = "non_axial_or_unknown_plane"
            continue
        if _is_derived(row):
            row["_selection_exclusion"] = "derived_or_secondary"
            continue
        if label == "FLAIR":
            valid, reason = _flair_valid(row, config)
            row["flair_validation"] = reason
            if not valid:
                row["_selection_exclusion"] = reason
                continue
        row["_effective_label"] = label
        candidates.append(row)
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in candidates:
        key = (text(row.get("tumor_category")), text(row.get("subject")))
        groups.setdefault(key, []).append(row)
    selected: list[dict[str, str]] = []
    eligible: list[dict[str, str]] = []
    for key, group in groups.items():
        study_groups: dict[str, list[dict[str, str]]] = {}
        for row in group:
            study_groups.setdefault(text(row.get("StudyInstanceUID")), []).append(row)
        if config.require_same_study:
            study_groups = {study: group_rows for study, group_rows in study_groups.items() if study}
        if not study_groups:
            continue
        study = max(study_groups, key=lambda name: (len({_effective_label(row) for row in study_groups[name]}), sum(int(_effective_label(row) in REQUIRED_MODALITIES) for row in study_groups[name])))
        study_rows = study_groups[study]
        chosen: dict[str, dict[str, str]] = {}
        for modality in REQUIRED_MODALITIES:
            options = [row for row in study_rows if _effective_label(row) == modality]
            if options:
                chosen[modality] = max(options, key=lambda row: _candidate_score(row, modality, config))
        if set(chosen) != set(REQUIRED_MODALITIES):
            continue
        for modality, row in chosen.items():
            row = dict(row)
            row.update({"selection_modality": modality, "selection_source": "same_study_axial_qc"})
            selected.append(row)
        eligible.append({"tumor_category": key[0], "subject": key[1], "StudyInstanceUID": study, "complete_modalities": ",".join(REQUIRED_MODALITIES)})
    return selected, eligible


def _stage_input(row: dict[str, str], staging_root: Path) -> tuple[Path, Path | None]:
    source = Path(text(row.get("source_folder_abs")))
    uid_clean = bool_value(row.get("source_folder_is_uid_clean"))
    if uid_clean and source.is_dir():
        return source, None
    filelist = Path(text(row.get("filelist_abs")))
    stage = Path(tempfile.mkdtemp(prefix=f"{text(row.get('record_id'))}_", dir=staging_root))
    if not filelist.is_file():
        raise FileNotFoundError(f"Series filelist is missing: {filelist}")
    for line in filelist.read_text(encoding="utf-8").splitlines():
        path = Path(line.strip())
        if path.is_file():
            shutil.copy2(path, stage / path.name)
    return stage, stage


def _run_dcm2niix(input_dir: Path, output_dir: Path, stem: str, config: ConversionConfig) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [config.dcm2niix, "-z", "y", "-f", stem, "-o", str(output_dir), "-b", "y", "-ba", "n", str(input_dir)]
    started = time.monotonic()
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=config.timeout_seconds, check=False)
    return {"command": command, "return_code": completed.returncode, "stdout": completed.stdout, "seconds": round(time.monotonic() - started, 3)}


def _resolve_output(output_dir: Path, stem: str) -> tuple[bool, str, str]:
    exact = output_dir / f"{stem}.nii.gz"
    if exact.is_file():
        return True, str(exact), "exact"
    matches = sorted(output_dir.glob(f"{stem}*.nii.gz"))
    if len(matches) == 1:
        return True, str(matches[0]), "unique_prefix"
    return False, "", "no_output" if not matches else "multiple_outputs"


def convert_one(row: dict[str, str], config: ConversionConfig) -> dict[str, Any]:
    modality = text(row.get("selection_modality") or row.get("_effective_label")).upper()
    category, subject = safe_name(row.get("tumor_category")), safe_name(row.get("subject"))
    subject_dir = config.output_dir / f"{category}-{subject}"
    subject_dir.mkdir(parents=True, exist_ok=True)
    stem = MODALITY_TO_FILENAME[modality]
    target = subject_dir / f"{stem}.nii.gz"
    result: dict[str, Any] = {"tumor_category": category, "subject": subject, "modality": modality, "record_id": text(row.get("record_id")), "nii_path": str(target), "status": "pending"}
    if config.resume and target.is_file() and target.stat().st_size >= config.min_nifti_file_size_bytes:
        result.update({"status": "skipped_existing", "dcm2niix_return_code": 0})
        return result
    staging_root = config.output_dir / "_series_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    input_dir, cleanup_dir = _stage_input(row, staging_root)
    try:
        log = _run_dcm2niix(input_dir, subject_dir, stem, config)
        ok, resolved, resolution = _resolve_output(subject_dir, stem)
        if not ok or log["return_code"] != 0:
            result.update({"status": "failed", "error": f"dcm2niix output={resolution}", "dcm2niix_return_code": log["return_code"], "dcm2niix_stdout": log["stdout"], "dcm2niix_seconds": log["seconds"]})
        else:
            if Path(resolved) != target:
                if target.exists():
                    target.unlink()
                Path(resolved).replace(target)
            result.update({"status": "converted", "nii_path": str(target), "dcm2niix_return_code": log["return_code"], "dcm2niix_stdout": log["stdout"], "dcm2niix_seconds": log["seconds"]})
    except subprocess.TimeoutExpired as exc:
        result.update({"status": "failed", "error": f"dcm2niix timeout after {config.timeout_seconds}s", "dcm2niix_stdout": str(exc.stdout or "")})
    except Exception as exc:
        result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
    return result


def nifti_qc_one(path: str, config: ConversionConfig) -> dict[str, Any]:
    result: dict[str, Any] = {"nii_path": path, "exists": Path(path).is_file(), "qc_pass": False}
    if not result["exists"]:
        result["qc_error"] = "file_missing"
        return result
    result["file_size_bytes"] = Path(path).stat().st_size
    if result["file_size_bytes"] < config.min_nifti_file_size_bytes:
        result["qc_error"] = "file_too_small"
        return result
    try:
        image = nib.load(path)
        shape = tuple(int(x) for x in image.shape)
        spacing = tuple(float(x) for x in image.header.get_zooms()[:3])
        affine = np.asarray(image.affine)
        result.update({"shape": "x".join(map(str, shape)), "ndim": len(shape), "voxel_spacing_mm": ",".join(f"{x:.6g}" for x in spacing), "affine": json.dumps(affine.tolist()), "slice_count": max(shape) if shape else 0})
        valid = len(shape) >= 3 and max(shape) >= config.min_valid_slices and np.isfinite(affine).all() and all(0 < value <= config.max_valid_voxel_spacing_mm for value in spacing)
        result["qc_pass"] = bool(valid)
        if not valid:
            result["qc_error"] = "shape_spacing_or_affine_invalid"
    except Exception as exc:
        result["qc_error"] = f"{type(exc).__name__}: {exc}"
    return result


def run_conversion(config: ConversionConfig) -> dict[str, str]:
    rows = _read_rows(config.manifest_csv)
    selected, eligible = select_four_modalities(rows, config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(eligible, config.output_dir / "eligible_four_modality_subjects.csv")
    atomic_csv(selected, config.output_dir / "selected_four_modality_series.csv")
    converted: list[dict[str, Any]] = []
    for row in tqdm(selected, desc="Converting selected series", unit="series"):
        converted.append(convert_one(row, config))
    atomic_csv(converted, config.output_dir / "conversion_summary.csv")
    qc_rows = [nifti_qc_one(row["nii_path"], config) for row in converted if row.get("nii_path")]
    atomic_csv(qc_rows, config.output_dir / "nifti_qc.csv")
    atomic_json({"schema_version": "1.0", "manifest_csv": str(config.manifest_csv.resolve()), "selected_series": len(selected), "eligible_subjects": len(eligible), "converted_rows": len(converted)}, config.output_dir / "conversion_config.json")
    return {"eligible": str(config.output_dir / "eligible_four_modality_subjects.csv"), "selected": str(config.output_dir / "selected_four_modality_series.csv"), "summary": str(config.output_dir / "conversion_summary.csv"), "nifti_qc": str(config.output_dir / "nifti_qc.csv")}
