from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from PIL import Image

from .utils import integer, natural_key, number, relative_path, text, stable_id


MISSING_UID = "__MISSING_SERIES_UID__"
METADATA_FIELDS = (
    "StudyInstanceUID", "SeriesInstanceUID", "StudyDescription", "SeriesDescription",
    "ProtocolName", "SequenceName", "ScanningSequence", "SequenceVariant", "ScanOptions",
    "ImageType", "MRAcquisitionType", "RepetitionTime", "EchoTime", "InversionTime",
    "FlipAngle", "DiffusionBValue", "ContrastBolusAgent", "ContrastBolusVolume", "Rows",
    "Columns", "SliceThickness", "PixelSpacing", "ImageOrientationPatient", "ImagePositionPatient",
    "Manufacturer", "MagneticFieldStrength", "PhotometricInterpretation",
)


@dataclass
class LogicalSeries:
    record_id: str
    tumor_category: str
    subject: str
    source_folder_abs: str
    source_folder_rel: str
    series_folder: str
    series_uid: str
    study_uid: str
    files: list[str]
    middle_dicom_abs: str
    middle_index: int
    metadata: dict[str, str]
    folder_uid_group_count: int
    uid_group_index: int

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "record_id": self.record_id,
            "tumor_category": self.tumor_category,
            "subject": self.subject,
            "source_folder_abs": self.source_folder_abs,
            "source_folder_rel": self.source_folder_rel,
            "series_folder": self.series_folder,
            "SeriesInstanceUID": self.series_uid,
            "StudyInstanceUID": self.study_uid,
            "n_dicoms": len(self.files),
            "middle_index_0based": self.middle_index,
            "middle_dicom_abs": self.middle_dicom_abs,
            "middle_dicom_rel": relative_path(self.middle_dicom_abs, self.source_folder_abs),
            "folder_uid_group_count": self.folder_uid_group_count,
            "uid_group_index": self.uid_group_index,
            "short_series": len(self.files) <= 3,
        }
        row.update(self.metadata)
        return row


def read_header(path: str | Path) -> Any:
    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)


def is_image_dicom(path: str | Path) -> bool:
    try:
        ds = read_header(path)
        return getattr(ds, "Rows", None) is not None and getattr(ds, "Columns", None) is not None
    except Exception:
        return False


def image_files(folder: str | Path, recursive: bool = False) -> list[str]:
    folder = Path(folder)
    paths = folder.rglob("*") if recursive else folder.iterdir()
    return sorted([str(path) for path in paths if path.is_file() and is_image_dicom(path)], key=lambda path: natural_key(Path(path).name))


def series_uid(path: str | Path) -> str:
    try:
        return text(getattr(read_header(path), "SeriesInstanceUID", "")) or MISSING_UID
    except Exception:
        return MISSING_UID


def group_by_series_uid(paths: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in paths:
        groups.setdefault(series_uid(path), []).append(path)
    return groups


def spatial_position(ds: Any) -> float | None:
    try:
        orientation = np.asarray(getattr(ds, "ImageOrientationPatient"), dtype=float)
        position = np.asarray(getattr(ds, "ImagePositionPatient"), dtype=float)
        if orientation.size != 6 or position.size != 3:
            return None
        normal = np.cross(orientation[:3], orientation[3:])
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            return None
        return float(np.dot(position, normal / norm))
    except Exception:
        return None


def sort_files(paths: list[str]) -> list[str]:
    info = []
    for path in paths:
        try:
            ds = read_header(path)
            spatial = spatial_position(ds)
            instance = integer(getattr(ds, "InstanceNumber", None))
        except Exception:
            spatial, instance = None, None
        info.append((path, spatial, instance))
    if sum(item[1] is not None for item in info) >= max(2, int(len(info) * 0.8)):
        return [item[0] for item in sorted(info, key=lambda item: item[1] if item[1] is not None else math.inf)]
    if sum(item[2] is not None for item in info) >= max(2, int(len(info) * 0.8)):
        return [item[0] for item in sorted(info, key=lambda item: item[2] if item[2] is not None else math.inf)]
    return sorted(paths, key=lambda path: natural_key(Path(path).name))


def metadata(path: str | Path) -> dict[str, str]:
    try:
        ds = read_header(path)
    except Exception:
        return {}
    result: dict[str, str] = {}
    for field in METADATA_FIELDS:
        value = getattr(ds, field, "")
        if isinstance(value, (list, tuple)):
            value = ",".join(text(item) for item in value)
        result[field] = text(value)
    return result


def resize_with_padding(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    scale = min(size / max(width, 1), size / max(height, 1))
    resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def dicom_to_pil(path: str | Path, image_size: int = 512) -> Image.Image:
    ds = pydicom.dcmread(str(path), force=True)
    try:
        array = np.asarray(ds.pixel_array)
    except Exception as exc:
        raise RuntimeError(f"Unable to decode DICOM pixels: {path}. Install a suitable compressed-DICOM decoder if needed.") from exc
    if array.ndim == 3 and not (array.shape[-1] == 3 and array.shape[0] > 4):
        array = array[array.shape[0] // 2]
    array = array.astype(np.float32)
    array = array * (number(getattr(ds, "RescaleSlope", 1), 1) or 1) + (number(getattr(ds, "RescaleIntercept", 0), 0) or 0)
    if array.ndim == 3 and array.shape[-1] == 3:
        low, high = np.nanpercentile(array, [0.5, 99.5])
        array = np.clip((array - low) / max(high - low, 1e-6), 0, 1)
        return resize_with_padding(Image.fromarray((array * 255).astype(np.uint8)).convert("RGB"), image_size)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise RuntimeError(f"DICOM contains no finite pixels: {path}")
    low, high = np.percentile(finite, [0.5, 99.5])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    array = np.clip((array - low) / max(high - low, 1e-6), 0, 1)
    if text(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        array = 1 - array
    return resize_with_padding(Image.fromarray((array * 255).astype(np.uint8)).convert("RGB"), image_size)


def collect_subject_series(subject_dir: str | Path, root_dir: str | Path, tumor_category: str, subject: str, filelist_dir: str | Path) -> list[LogicalSeries]:
    subject_dir, filelist_dir = Path(subject_dir), Path(filelist_dir)
    filelist_dir.mkdir(parents=True, exist_ok=True)
    records: list[LogicalSeries] = []
    for folder in sorted((path for path in subject_dir.iterdir() if path.is_dir()), key=lambda path: natural_key(path.name)):
        groups = group_by_series_uid(image_files(folder))
        for uid_index, (uid, paths) in enumerate(sorted(groups.items()), start=1):
            ordered = sort_files(paths)
            if not ordered:
                continue
            middle_index, middle = len(ordered) // 2, ordered[len(ordered) // 2]
            meta = metadata(middle)
            actual_uid = meta.get("SeriesInstanceUID") or uid
            record_id = stable_id(tumor_category, subject, relative_path(folder, root_dir), actual_uid)
            filelist = filelist_dir / f"{record_id}.txt"
            filelist.write_text("\n".join(os.path.abspath(path) for path in ordered) + "\n", encoding="utf-8")
            meta["filelist_abs"] = str(filelist.resolve())
            records.append(LogicalSeries(
                record_id=record_id, tumor_category=tumor_category, subject=subject,
                source_folder_abs=str(folder.resolve()), source_folder_rel=relative_path(folder, root_dir),
                series_folder=folder.name, series_uid=actual_uid, study_uid=meta.get("StudyInstanceUID", ""),
                files=ordered, middle_dicom_abs=str(Path(middle).resolve()), middle_index=middle_index,
                metadata=meta, folder_uid_group_count=len(groups), uid_group_index=uid_index,
            ))
    return records
