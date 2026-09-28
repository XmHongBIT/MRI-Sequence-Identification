from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import atomic_csv, atomic_json, bool_value, number, natural_key, safe_name, text


MODALITY_TO_FILENAME = {
    "T1": "t1",
    "T1CE": "t1ce",
    "T2": "t2",
    "FLAIR": "t2flair",
}
MODALITY_OUTPUT_ORDER = ("T1", "T1CE", "T2", "FLAIR")
MODALITY_SELECT_ORDER = ("FLAIR", "T1", "T1CE", "T2")
MODALITY_COUNT_COLUMN = {
    "T1": "T1_count",
    "T1CE": "T1CE_count",
    "T2": "T2_count",
    "FLAIR": "FLAIR_model_count",
}
T1_KEYWORDS = ("MPRAGE", "BRAVO", "SPGR", "T1W", "T1 W")
FLAIR_KEYWORDS = ("FLAIR", "T2FLAIR", "T2_FLAIR", "T2-FLAIR")
T1CE_REJECT_KEYWORDS = ("TOF", "MRA", "ANGIO")
T1CE_CONTRAST_KEYWORDS = (
    "+C", "+G", "C+", "POST", "POSTCONTRAST", "POST-CONTRAST", "CONTRAST",
    "GAD", "GD", "IR+C", "T1C", "T1CE",
)


@dataclass
class ConversionConfig:
    manifest_csv: Path
    output_dir: Path
    dcm2niix: str = "dcm2niix"
    workers: int = 4
    timeout_seconds: int = 600
    resume: bool = True
    require_same_study: bool = True
    axial_cos_threshold: float = 0.85
    allow_unknown_plane: bool = False
    strict_flair_validation: bool = True
    flair_metadata_rescue: bool = True
    flair_min_ti_ms: float = 1500.0
    flair_min_te_ms: float = 50.0
    flair_min_tr_ms: float = 4000.0
    t1_metadata_rescue: bool = True
    flair_contradiction_te_max_ms: float = 30.0
    flair_contradiction_ti_max_ms: float = 1500.0
    t1ce_metadata_rescue: bool = True
    t1ce_require_contrast_evidence: bool = False
    t1ce_t1like_te_max_ms: float = 30.0
    selected_modalities: tuple[str, ...] = MODALITY_OUTPUT_ORDER
    required_modalities: frozenset[str] = frozenset()
    min_modalities_per_subject: int = 1
    path_prefix_remap: tuple[tuple[str, str], ...] = ()
    preflight_min_reachable_fraction: float = 0.5
    preflight_sample_size: int = 300
    diagnose_only: bool = False
    min_nifti_file_size_bytes: int = 10 * 1024
    min_valid_slices: int = 5
    max_valid_voxel_spacing_mm: float = 10.0

    def __post_init__(self) -> None:
        self.manifest_csv = Path(self.manifest_csv)
        self.output_dir = Path(self.output_dir)
        selected = tuple(dict.fromkeys(item.upper() for item in self.selected_modalities))
        unknown = set(selected) - set(MODALITY_TO_FILENAME)
        if unknown:
            raise ValueError(f"Unknown modalities: {sorted(unknown)}")
        required = frozenset(item.upper() for item in self.required_modalities)
        if not required.issubset(set(selected)):
            raise ValueError("required_modalities must be a subset of selected_modalities")
        if self.min_modalities_per_subject < 0:
            raise ValueError("min_modalities_per_subject must be non-negative")
        self.selected_modalities = selected
        self.required_modalities = required

    @property
    def selected_output_order(self) -> list[str]:
        head = [item for item in MODALITY_OUTPUT_ORDER if item in self.selected_modalities]
        tail = [item for item in self.selected_modalities if item not in MODALITY_OUTPUT_ORDER]
        return head + tail

    @property
    def selected_select_order(self) -> list[str]:
        head = [item for item in MODALITY_SELECT_ORDER if item in self.selected_modalities]
        tail = [item for item in self.selected_modalities if item not in MODALITY_SELECT_ORDER]
        return head + tail


def _atomic_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def _remap_path(path: Any, config: ConversionConfig) -> str:
    value = text(path)
    for source, target in config.path_prefix_remap:
        if value.startswith(source):
            return target + value[len(source):]
    return value


def _read_header(path: str) -> Any:
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("pydicom is required for DICOM QC. Install the project dependencies first.") from exc
    return pydicom.dcmread(path, stop_before_pixels=True, force=True)


def _filelist_paths(path: Any, config: ConversionConfig) -> list[str]:
    filelist = _remap_path(path, config)
    if not filelist or not os.path.isfile(filelist):
        return []
    paths: list[str] = []
    try:
        for line in Path(filelist).read_text(encoding="utf-8").splitlines():
            candidate = _remap_path(line, config)
            if candidate and os.path.isfile(candidate):
                paths.append(candidate)
    except OSError:
        return []
    return paths


def _find_sample_dicom(row: dict[str, Any], config: ConversionConfig) -> str:
    filelist_paths = _filelist_paths(row.get("filelist_abs", ""), config)
    if filelist_paths:
        return filelist_paths[0]
    folder = _remap_path(row.get("source_folder_abs", ""), config)
    if not os.path.isdir(folder):
        return ""
    try:
        for name in sorted(os.listdir(folder), key=natural_key):
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            try:
                ds = _read_header(path)
                if getattr(ds, "Rows", None) is not None:
                    return path
            except Exception:
                continue
    except OSError:
        pass
    return ""


def _get_iop(ds: Any) -> list[float] | None:
    iop = getattr(ds, "ImageOrientationPatient", None)
    try:
        if iop is not None and len(iop) == 6:
            return [float(value) for value in iop]
    except Exception:
        pass
    try:
        sequence = ds.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0]
        iop = sequence.ImageOrientationPatient
        if len(iop) == 6:
            return [float(value) for value in iop]
    except Exception:
        return None
    return None


def _classify_plane(iop: list[float] | None, threshold: float) -> dict[str, Any]:
    if iop is None or len(iop) != 6:
        return {"plane": "UNKNOWN", "normal_x": np.nan, "normal_y": np.nan, "normal_z": np.nan, "plane_cosine": np.nan}
    row = np.asarray(iop[:3], dtype=np.float64)
    col = np.asarray(iop[3:], dtype=np.float64)
    normal = np.cross(row, col)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return {"plane": "UNKNOWN", "normal_x": np.nan, "normal_y": np.nan, "normal_z": np.nan, "plane_cosine": np.nan}
    normal = normal / norm
    axis = int(np.argmax(np.abs(normal)))
    cosine = float(abs(normal[axis]))
    if cosine < threshold:
        plane = "OBLIQUE"
    else:
        plane = {0: "SAGITTAL", 1: "CORONAL", 2: "AXIAL"}[axis]
    return {"plane": plane, "normal_x": float(normal[0]), "normal_y": float(normal[1]), "normal_z": float(normal[2]), "plane_cosine": cosine}


def _enrich_row(row: dict[str, Any], config: ConversionConfig) -> dict[str, Any]:
    record = dict(row)
    sample_path = _find_sample_dicom(record, config)
    record["qc_sample_dicom"] = sample_path
    record.update(_classify_plane(None, config.axial_cos_threshold))
    if not sample_path:
        return record
    try:
        ds = _read_header(sample_path)
        record.update(_classify_plane(_get_iop(ds), config.axial_cos_threshold))
        for field in ("RepetitionTime", "EchoTime", "InversionTime", "SeriesDescription", "ProtocolName", "ImageType", "ContrastBolusAgent"):
            value = getattr(ds, field, None)
            if value is not None and text(value):
                record[field] = text(value)
    except Exception as exc:
        record["orientation_qc_error"] = f"{type(exc).__name__}: {exc}"
    return record


def enrich_manifest_with_dicom_qc(df: pd.DataFrame, config: ConversionConfig) -> pd.DataFrame:
    records = [_enrich_row(row.to_dict(), config) for _, row in tqdm(df.iterrows(), total=len(df), desc="Reading DICOM orientation", unit="series", dynamic_ncols=True)]
    return pd.DataFrame(records)


def _safe_mask(df: pd.DataFrame, predicate: Any) -> pd.Series:
    if df.empty:
        return pd.Series(False, index=df.index, dtype=bool)
    return pd.Series([bool(predicate(row)) for _, row in df.iterrows()], index=df.index, dtype=bool)


def _numeric(row: Any, key: str) -> float:
    value = row.get(key, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _description_text(row: Any) -> str:
    # Do not inspect ProtocolName for sequence-name decisions. Some sites use a
    # protocol name containing "MRA" for ordinary brain MRI examinations.
    return text(row.get("SeriesDescription", "")).upper()


def _contains_any(value: Any, keys: Iterable[str]) -> bool:
    upper = text(value).upper()
    return any(key.upper() in upper for key in keys)


def _has_t1_token(value: Any) -> bool:
    return re.search(r"(?<![A-Z0-9])T1(?!C|E)", text(value).upper()) is not None


def _has_t2_token(value: Any) -> bool:
    return re.search(r"(?<![A-Z0-9])T2", text(value).upper()) is not None


def _is_axial(row: Any, config: ConversionConfig) -> bool:
    plane = text(row.get("plane", "")).upper()
    return plane == "AXIAL" or (config.allow_unknown_plane and plane == "UNKNOWN")


def _flair_features(row: Any, config: ConversionConfig) -> dict[str, Any]:
    full_text = " ".join(text(row.get(key, "")) for key in ("SeriesDescription", "ProtocolName", "ImageType")).upper()
    description = _description_text(row)
    tr, te, ti = _numeric(row, "RepetitionTime"), _numeric(row, "EchoTime"), _numeric(row, "InversionTime")
    explicit_flair = _contains_any(full_text, FLAIR_KEYWORDS)
    explicit_t1 = _contains_any(full_text, T1_KEYWORDS)
    physical_flair = not np.isnan(ti) and not np.isnan(te) and ti >= config.flair_min_ti_ms and te >= config.flair_min_te_ms and (np.isnan(tr) or tr >= config.flair_min_tr_ms)
    t1_like_parameters = not np.isnan(te) and te < 25.0 and (np.isnan(ti) or ti < config.flair_min_ti_ms) and (np.isnan(tr) or tr < config.flair_min_tr_ms)
    t1_token, t2_token = _has_t1_token(description), _has_t2_token(description)
    t1_like_token = t1_token and not t2_token
    t1_like_physics = (not np.isnan(te) and te < config.flair_contradiction_te_max_ms) or (not np.isnan(ti) and ti < config.flair_contradiction_ti_max_ms)
    return {
        "explicit_flair": explicit_flair, "explicit_t1": explicit_t1, "physical_flair": physical_flair,
        "t1_like_parameters": t1_like_parameters, "has_t1_token": t1_token, "has_t2_token": t2_token,
        "t1_like_token": t1_like_token, "t1_like_physics": t1_like_physics,
        "t1_contradicts_flair": t1_like_token or explicit_t1,
    }


def _flair_is_plausible(row: Any, config: ConversionConfig) -> bool:
    features = _flair_features(row, config)
    if config.t1_metadata_rescue and features["t1_contradicts_flair"]:
        return False
    if features["explicit_flair"] or features["physical_flair"]:
        return True
    if not config.strict_flair_validation:
        return (
            text(row.get("final_label", "")).upper() == "FLAIR"
            and not features["t1_like_parameters"]
            and not (config.t1_metadata_rescue and features["t1_contradicts_flair"])
        )
    return False


def _t1_features(row: Any, config: ConversionConfig) -> dict[str, Any]:
    description = _description_text(row)
    te, ti = _numeric(row, "EchoTime"), _numeric(row, "InversionTime")
    t1_token, t2_token = _has_t1_token(description), _has_t2_token(description)
    return {
        "has_t1_token": t1_token, "has_t2_token": t2_token,
        "explicit_t1": _contains_any(description, T1_KEYWORDS),
        "t1_like_token": t1_token and not t2_token,
        "t1_like_physics": (not np.isnan(te) and te < config.flair_contradiction_te_max_ms) or (not np.isnan(ti) and ti < config.flair_min_ti_ms),
    }


def _t1_is_plausible(row: Any, config: ConversionConfig) -> bool:
    features = _t1_features(row, config)
    if features["has_t2_token"] and not features["has_t1_token"]:
        return False
    return bool(features["t1_like_token"] or features["explicit_t1"])


def _t1ce_features(row: Any, config: ConversionConfig) -> dict[str, Any]:
    description = _description_text(row)
    te = _numeric(row, "EchoTime")
    contrast_agent = text(row.get("ContrastBolusAgent", ""))
    is_angio = _contains_any(description, T1CE_REJECT_KEYWORDS)
    contrast_hint = bool(contrast_agent) or _contains_any(description, T1CE_CONTRAST_KEYWORDS)
    t1_like = (not np.isnan(te) and te < config.t1ce_t1like_te_max_ms) or _contains_any(description, T1_KEYWORDS) or _has_t1_token(description) or _contains_any(description, ("IR", "DARK-FLUID", "DARKFLUID", "TIRM"))
    return {"is_angio": is_angio, "contrast_hint": contrast_hint, "t1_like": t1_like}


def _t1ce_is_acceptable(row: Any, config: ConversionConfig) -> bool:
    features = _t1ce_features(row, config)
    if features["is_angio"]:
        return False
    return not (config.t1ce_require_contrast_evidence and not features["contrast_hint"])


def _t1ce_is_plausible(row: Any, config: ConversionConfig) -> bool:
    features = _t1ce_features(row, config)
    return not features["is_angio"] and features["contrast_hint"] and features["t1_like"]


def _common_score(row: Any) -> float:
    score = 0.0
    if bool_value(row.get("dcm2nii_ready_direct", False)):
        score += 20
    if bool_value(row.get("source_folder_is_uid_clean", False)):
        score += 10
    confidence = text(row.get("model_confidence", "")).upper()
    score += {"HIGH": 10, "MEDIUM": 5}.get(confidence, 0)
    try:
        score += min(float(row.get("n_dicoms", 0)) / 20.0, 10.0)
    except (TypeError, ValueError):
        pass
    description = " ".join(text(row.get(key, "")) for key in ("SeriesDescription", "ProtocolName", "ImageType"))
    if _contains_any(description, ("MPRAGE", "BRAVO", "SPGR", "SPACE", "CUBE")):
        score += 5
    if _contains_any(description, ("LOCALIZER", "SCOUT", "DERIVED", "SECONDARY")):
        score -= 50
    return score


def _flair_score(row: Any, config: ConversionConfig) -> float:
    score = _common_score(row)
    features = _flair_features(row, config)
    if text(row.get("final_label", "")).upper() == "FLAIR":
        score += 40
    if features["explicit_flair"]:
        score += 100
    if features["physical_flair"]:
        score += 80
    if features["explicit_t1"]:
        score -= 100
    if features["t1_like_parameters"]:
        score -= 120
    if features["t1_contradicts_flair"]:
        score -= 200
    return score


def _t1_score(row: Any, config: ConversionConfig) -> float:
    score = _common_score(row)
    features = _t1_features(row, config)
    label = text(row.get("final_label", "")).upper()
    if label == "T1":
        score += 40
    if features["t1_like_token"]:
        score += 60
    if features["explicit_t1"]:
        score += 20
    if features["t1_like_physics"]:
        score += 10
    if features["has_t2_token"]:
        score -= 80
    return score


def _t1ce_score(row: Any, config: ConversionConfig) -> float:
    score = _common_score(row)
    features = _t1ce_features(row, config)
    if text(row.get("ContrastBolusAgent", "")):
        score += 30
    if _contains_any(_description_text(row), ("POST", "POSTCONTRAST", "POST-CONTRAST")):
        score += 20
    if _contains_any(_description_text(row), ("+C", "+G", "GAD", "CONTRAST", "T1CE")):
        score += 15
    if features["contrast_hint"]:
        score += 25
    if features["t1_like"]:
        score += 15
    if features["is_angio"]:
        score -= 200
    return score


def _is_acceptable_original(row: Any, config: ConversionConfig) -> bool:
    image_type = text(row.get("ImageType", "")).upper()
    description = (text(row.get("SeriesDescription", "")) + " " + text(row.get("ProtocolName", ""))).upper()
    if any(token in image_type for token in ("DERIVED", "SECONDARY", "CSA RESAMPLED")) or "POSDISP" in description:
        return False
    if bool_value(row.get("short_series", False)):
        return False
    try:
        if float(row.get("n_dicoms", 0)) < config.min_valid_slices:
            return False
    except (TypeError, ValueError):
        pass
    return True


def _select_primary(group: pd.DataFrame, modality: str, config: ConversionConfig, exclude_record_ids: set[str]) -> dict[str, Any] | None:
    axial = group[_safe_mask(group, lambda row: _is_axial(row, config))].copy()
    if axial.empty:
        return None
    if "record_id" in axial.columns:
        axial = axial[~axial["record_id"].fillna("").astype(str).isin(exclude_record_ids)].copy()
    if axial.empty:
        return None
    if "n_dicoms" in axial.columns:
        axial["_n_dicoms"] = pd.to_numeric(axial["n_dicoms"], errors="coerce").fillna(0)
    else:
        axial["_n_dicoms"] = 0
    labels = axial["final_label"].fillna("").astype(str).str.upper()

    if modality == "FLAIR":
        candidates = axial[labels == "FLAIR"].copy()
        if not candidates.empty:
            candidates = candidates[_safe_mask(candidates, lambda row: _flair_is_plausible(row, config))].copy()
        if candidates.empty and config.flair_metadata_rescue:
            candidates = axial[_safe_mask(axial, lambda row: _flair_is_plausible(row, config))].copy()
            candidates = candidates[~candidates["final_label"].isin({"T1CE", "DWI", "ADC", "SWI", "LOCALIZER", "OTHER"})].copy()
        scorer, source_model, source_rescue = lambda row: _flair_score(row, config), "model_FLAIR_validated", "metadata_FLAIR_rescue"
        source_key = lambda row: source_model if text(row.get("final_label", "")).upper() == "FLAIR" else source_rescue
    elif modality == "T1":
        candidates = axial[labels == "T1"].copy()
        if candidates.empty and config.t1_metadata_rescue:
            candidates = axial[_safe_mask(axial, lambda row: _t1_is_plausible(row, config))].copy()
            candidates = candidates[~candidates["final_label"].isin({"T1CE", "DWI", "ADC", "SWI", "LOCALIZER", "OTHER"})].copy()
        scorer, source_model, source_rescue = lambda row: _t1_score(row, config), "model_T1_axial", "metadata_T1_rescue"
        source_key = lambda row: source_model if text(row.get("final_label", "")).upper() == "T1" else source_rescue
    elif modality == "T1CE":
        candidates = axial[labels == "T1CE"].copy()
        if not candidates.empty:
            candidates = candidates[_safe_mask(candidates, lambda row: _t1ce_is_acceptable(row, config))].copy()
        if candidates.empty and config.t1ce_metadata_rescue:
            candidates = axial[_safe_mask(axial, lambda row: _t1ce_is_plausible(row, config))].copy()
            candidates = candidates[~candidates["final_label"].isin({"DWI", "ADC", "SWI", "LOCALIZER", "OTHER"})].copy()
        scorer, source_model, source_rescue = lambda row: _t1ce_score(row, config), "model_T1CE_validated", "metadata_T1CE_rescue"
        source_key = lambda row: source_model if text(row.get("final_label", "")).upper() == "T1CE" else source_rescue
    else:
        candidates = axial[labels == modality].copy()
        scorer, source_model, source_rescue = _common_score, "model_label_axial", "model_label_axial"
        source_key = lambda row: source_model

    if candidates.empty:
        return None
    candidates = candidates.copy()
    candidates["_score"] = candidates.apply(scorer, axis=1)
    candidates["_selection_source"] = candidates.apply(source_key, axis=1)
    candidates = candidates.sort_values(["_score", "_n_dicoms"], ascending=[False, False], kind="stable")
    return candidates.iloc[0].to_dict()


def _study_quality_score(group: pd.DataFrame, config: ConversionConfig) -> float:
    return len(group) * 0.1 + sum(2.0 for _, row in group.iterrows() if _is_acceptable_original(row, config))


def build_selection_tables(df: pd.DataFrame, config: ConversionConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if "final_label" not in df.columns:
        df["final_label"] = df.get("model_prediction", "")
    df["final_label"] = df["final_label"].fillna("").astype(str).str.strip().str.upper()
    for column in ("tumor_category", "subject", "StudyInstanceUID"):
        if column not in df.columns:
            df[column] = ""
    eligible_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    subject_groups = df.groupby(["tumor_category", "subject"], sort=False, dropna=False)
    for (tumor, subject), subject_group in subject_groups:
        complete_studies: list[dict[str, Any]] = []
        for study_uid, study_group in subject_group.groupby("StudyInstanceUID", sort=False, dropna=False):
            study_uid = text(study_uid)
            if config.require_same_study and not study_uid:
                qc_rows.append({"tumor_category": tumor, "subject": subject, "StudyInstanceUID": "", "eligible": False, "reason": "missing_StudyInstanceUID"})
                continue
            clean_group = study_group[_safe_mask(study_group, lambda row: _is_acceptable_original(row, config))].copy()
            axial_group = clean_group[_safe_mask(clean_group, lambda row: _is_axial(row, config))].copy()
            if "qc_sample_dicom" in clean_group.columns:
                unavailable = int(clean_group["qc_sample_dicom"].fillna("").astype(str).str.strip().eq("").sum())
            else:
                unavailable = -1
            qc_row: dict[str, Any] = {
                "tumor_category": tumor, "subject": subject, "StudyInstanceUID": study_uid,
                "num_series_in_study": len(study_group), "num_clean_series": len(clean_group),
                "num_axial_series": len(axial_group), "num_nonaxial_clean_series": max(len(clean_group) - len(axial_group), 0),
                "num_orientation_unavailable": unavailable,
                "no_readable_dicom": bool(len(clean_group) > 0 and unavailable == len(clean_group)),
                "axial_FLAIR_plausible_count": int(_safe_mask(axial_group, lambda row: _flair_is_plausible(row, config)).sum()) if len(axial_group) else 0,
            }
            for modality in config.selected_output_order:
                qc_row[f"axial_{modality}_model_count"] = int((axial_group["final_label"] == modality).sum()) if "final_label" in axial_group else 0
            chosen: dict[str, dict[str, Any]] = {}
            used_record_ids: set[str] = set()
            missing: list[str] = []
            for modality in config.selected_select_order:
                selected = _select_primary(clean_group, modality, config, used_record_ids)
                if selected is None:
                    missing.append(modality)
                    continue
                record_id = text(selected.get("record_id", ""))
                if record_id:
                    used_record_ids.add(record_id)
                chosen[modality] = selected
            missing_required = [modality for modality in missing if modality in config.required_modalities]
            if missing_required:
                qc_row.update({"eligible": False, "reason": "missing_required:" + ",".join(missing_required)})
                qc_rows.append(qc_row)
                continue
            if len(chosen) < config.min_modalities_per_subject:
                qc_row.update({"eligible": False, "reason": f"too_few_modalities:{len(chosen)}<{config.min_modalities_per_subject}"})
                qc_rows.append(qc_row)
                continue
            qc_row.update({"eligible": True, "reason": ""})
            qc_rows.append(qc_row)
            complete_studies.append({"study_uid": study_uid, "study_group": study_group, "axial_group": axial_group, "chosen": chosen, "study_score": _study_quality_score(study_group, config)})
        if not complete_studies:
            continue
        best = max(complete_studies, key=lambda item: item["study_score"])
        chosen = best["chosen"]
        axial_group = best["axial_group"]
        study_uid = best["study_uid"]
        subject_row: dict[str, Any] = {
            "tumor_category": tumor, "subject": subject, "StudyInstanceUID": study_uid,
            "complete_study_count": len(complete_studies), "selected_study_score": best["study_score"],
            "num_series_in_selected_study": len(best["study_group"]), "num_axial_series": len(axial_group),
            "modalities_selected": ",".join(config.selected_output_order),
            "modality_absent": ",".join(modality for modality in config.selected_output_order if modality not in chosen),
        }
        for modality in config.selected_output_order:
            column = MODALITY_COUNT_COLUMN.get(modality, f"{modality}_count")
            subject_row[column] = int((axial_group["final_label"] == modality).sum()) if "final_label" in axial_group else 0
        for modality in config.selected_output_order:
            selected = chosen.get(modality)
            if selected is None:
                continue
            flair = _flair_features(selected, config) if modality == "FLAIR" else {}
            t1 = _t1_features(selected, config) if modality == "T1" else {}
            t1ce = _t1ce_features(selected, config) if modality == "T1CE" else {}
            row = {
                "tumor_category": tumor, "subject": subject, "StudyInstanceUID": study_uid, "modality": modality,
                "output_filename": MODALITY_TO_FILENAME[modality] + ".nii.gz", "record_id": text(selected.get("record_id", "")),
                "SeriesInstanceUID": text(selected.get("SeriesInstanceUID", "")), "source_folder_abs": text(selected.get("source_folder_abs", "")),
                "source_folder_rel": text(selected.get("source_folder_rel", "")), "filelist_abs": text(selected.get("filelist_abs", "")),
                "source_folder_is_uid_clean": text(selected.get("source_folder_is_uid_clean", "")), "dcm2nii_ready_direct": text(selected.get("dcm2nii_ready_direct", "")),
                "n_dicoms": text(selected.get("n_dicoms", "")), "selection_score": float(selected.get("_score", 0.0)),
                "selection_source": text(selected.get("_selection_source", "")), "plane": text(selected.get("plane", "")),
                "plane_cosine": text(selected.get("plane_cosine", "")), "SeriesDescription": text(selected.get("SeriesDescription", "")),
                "ProtocolName": text(selected.get("ProtocolName", "")), "SequenceName": text(selected.get("SequenceName", "")),
                "ImageType": text(selected.get("ImageType", "")), "RepetitionTime": text(selected.get("RepetitionTime", "")),
                "EchoTime": text(selected.get("EchoTime", "")), "InversionTime": text(selected.get("InversionTime", "")),
                "ContrastBolusAgent": text(selected.get("ContrastBolusAgent", "")), "ContrastBolusVolume": text(selected.get("ContrastBolusVolume", "")),
                "model_original_label": text(selected.get("final_label", "")), "model_confidence": text(selected.get("model_confidence", "")),
                "model_reason": text(selected.get("model_reason", "")), "flair_explicit_keyword": flair.get("explicit_flair", ""),
                "flair_physical_parameters": flair.get("physical_flair", ""), "flair_t1_like_parameters": flair.get("t1_like_parameters", ""),
                "flair_t1_contradiction": flair.get("t1_contradicts_flair", ""), "t1_has_t1_token": t1.get("has_t1_token", ""),
                "t1_has_t2_token": t1.get("has_t2_token", ""), "t1_like_token": t1.get("t1_like_token", ""),
                "t1ce_is_angio": t1ce.get("is_angio", ""), "t1ce_contrast_hint": t1ce.get("contrast_hint", ""),
                "t1ce_t1_like": t1ce.get("t1_like", ""),
            }
            selected_rows.append(row)
            prefix = "T2FLAIR" if modality == "FLAIR" else modality
            subject_row[f"{prefix}_record_id"] = row["record_id"]
            subject_row[f"{prefix}_source_folder"] = row["source_folder_abs"]
            subject_row[f"{prefix}_selection_source"] = row["selection_source"]
        eligible_rows.append(subject_row)
    return pd.DataFrame(eligible_rows), pd.DataFrame(selected_rows), pd.DataFrame(qc_rows)


def _prepare_input(row: dict[str, Any], config: ConversionConfig, staging_root: Path) -> tuple[str, Path | None]:
    source_folder = _remap_path(row.get("source_folder_abs", ""), config)
    if bool_value(row.get("dcm2nii_ready_direct", False)) and bool_value(row.get("source_folder_is_uid_clean", False)) and os.path.isdir(source_folder):
        return source_folder, None
    files = _filelist_paths(row.get("filelist_abs", ""), config)
    if not files:
        raise RuntimeError(f"Selected series filelist is unavailable: record_id={row.get('record_id', '')}")
    staging_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=safe_name(row.get("record_id", "series")) + "_", dir=staging_root))
    for index, source in enumerate(files):
        shutil.copy2(source, stage / f"{index:06d}{Path(source).suffix or '.dcm'}")
    return str(stage), stage


def _run_dcm2niix(input_folder: str, output_dir: Path, stem: str, config: ConversionConfig) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [config.dcm2niix, "-z", "y", "-b", "n", "-f", stem, "-o", str(output_dir), input_folder]
    started = time.monotonic()
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=config.timeout_seconds, check=False)
        return {"return_code": result.returncode, "timed_out": False, "stdout": result.stdout, "seconds": round(time.monotonic() - started, 3)}
    except subprocess.TimeoutExpired as exc:
        return {"return_code": -999, "timed_out": True, "stdout": text(exc.stdout), "seconds": round(time.monotonic() - started, 3)}


def _resolve_output(output_dir: Path, stem: str) -> tuple[bool, str, str]:
    target = output_dir / f"{stem}.nii.gz"
    if target.is_file() and target.stat().st_size > 0:
        return True, str(target), "exact"
    matches = sorted(path for path in output_dir.glob(f"{stem}*.nii.gz") if path.is_file() and path.stat().st_size > 0)
    if len(matches) == 1:
        if matches[0] != target:
            matches[0].replace(target)
        return True, str(target), "single_output_renamed"
    return False, "", "multiple_outputs" if len(matches) > 1 else "no_output"


def _absent_record(tumor: str, subject: str, modality: str) -> dict[str, Any]:
    return {"tumor_category": tumor, "subject": subject, "modality": modality, "record_id": "", "SeriesInstanceUID": "", "source_folder_abs": "", "filelist_abs": "", "n_dicoms": "", "selection_score": "", "nii_path": "", "status": "absent", "dcm2niix_seconds": "", "dcm2niix_return_code": "", "output_resolution": "", "error": ""}


def _convert_subject(key: tuple[str, str], selected_group: pd.DataFrame, config: ConversionConfig) -> list[dict[str, Any]]:
    tumor, subject = key
    output_dir = config.output_dir / f"{safe_name(tumor)}-{safe_name(subject)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_root = config.output_dir / "_series_staging"
    log: dict[str, Any] = {"tumor_category": tumor, "subject": subject, "started_at": datetime.now().isoformat(timespec="seconds"), "output_dir": str(output_dir), "modalities": {}}
    rows: list[dict[str, Any]] = []
    for modality in config.selected_output_order:
        part = selected_group[selected_group["modality"] == modality]
        if part.empty:
            record = _absent_record(tumor, subject, modality)
            rows.append(record)
            log["modalities"][modality] = record
            continue
        if len(part) != 1:
            raise RuntimeError(f"Expected at most one selected {modality} series for {tumor}/{subject}; got {len(part)}")
        selected = part.iloc[0].to_dict()
        stem = MODALITY_TO_FILENAME[modality]
        target = output_dir / f"{stem}.nii.gz"
        record: dict[str, Any] = {"tumor_category": tumor, "subject": subject, "modality": modality, "record_id": text(selected.get("record_id", "")), "SeriesInstanceUID": text(selected.get("SeriesInstanceUID", "")), "source_folder_abs": _remap_path(selected.get("source_folder_abs", ""), config), "filelist_abs": _remap_path(selected.get("filelist_abs", ""), config), "n_dicoms": text(selected.get("n_dicoms", "")), "selection_score": text(selected.get("selection_score", "")), "nii_path": str(target), "status": "", "dcm2niix_seconds": "", "dcm2niix_return_code": "", "output_resolution": "", "error": ""}
        if config.resume and target.is_file() and target.stat().st_size > 0:
            record["status"] = "existing"
            rows.append(record)
            log["modalities"][modality] = record
            continue
        stage: Path | None = None
        try:
            input_folder, stage = _prepare_input(selected, config, staging_root)
            result = _run_dcm2niix(input_folder, output_dir, stem, config)
            ok, resolved, resolution = _resolve_output(output_dir, stem)
            record.update({"dcm2niix_seconds": result["seconds"], "dcm2niix_return_code": result["return_code"], "output_resolution": resolution})
            if result["timed_out"]:
                record.update({"status": "timeout", "error": f"dcm2niix exceeded {config.timeout_seconds} seconds"})
            elif ok and result["return_code"] == 0:
                record.update({"status": "success", "nii_path": resolved})
            else:
                record.update({"status": "failed", "error": f"No unique output: {resolution}"})
            log["modalities"][modality] = {**record, "dcm2niix_stdout": result["stdout"]}
        except Exception as exc:
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            log["modalities"][modality] = {**record, "traceback": traceback.format_exc()}
        finally:
            if stage is not None:
                shutil.rmtree(stage, ignore_errors=True)
        rows.append(record)
    convertible = [row for row in rows if row["status"] != "absent"]
    log["subject_status"] = "empty" if not convertible else ("complete" if all(row["status"] in {"success", "existing"} for row in convertible) else "incomplete_conversion")
    log["finished_at"] = datetime.now().isoformat(timespec="seconds")
    atomic_json(log, output_dir / "conversion_log.json")
    return rows


def nifti_qc_one(path: str, config: ConversionConfig) -> dict[str, Any]:
    result: dict[str, Any] = {"nii_path": path, "exists": False, "file_size_bytes": 0, "shape": "", "ndim": "", "voxel_spacing": "", "orientation": "", "det_affine": "", "qc_pass": False, "qc_reason": ""}
    if not path or not os.path.isfile(path):
        result["qc_reason"] = "missing_file"
        return result
    result["exists"] = True
    result["file_size_bytes"] = os.path.getsize(path)
    if result["file_size_bytes"] < config.min_nifti_file_size_bytes:
        result["qc_reason"] = "file_too_small"
        return result
    try:
        import nibabel as nib
        image = nib.load(path)
        shape = tuple(int(value) for value in image.shape)
        zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
        result.update({"shape": "x".join(str(value) for value in shape), "ndim": len(shape), "voxel_spacing": ",".join(f"{value:.6g}" for value in zooms), "orientation": "".join(nib.aff2axcodes(image.affine)), "det_affine": float(np.linalg.det(image.affine[:3, :3]))})
        reasons: list[str] = []
        if len(shape) < 3:
            reasons.append("ndim_lt_3")
        elif min(shape[:3]) < config.min_valid_slices:
            reasons.append("too_few_slices")
        if any(not np.isfinite(value) or value <= 0 or value > config.max_valid_voxel_spacing_mm for value in zooms):
            reasons.append("abnormal_spacing")
        if not np.isfinite(image.affine).all():
            reasons.append("invalid_affine")
        if abs(result["det_affine"]) < 1e-8:
            reasons.append("singular_affine")
        result["qc_reason"] = ",".join(reasons)
        result["qc_pass"] = not reasons
    except Exception as exc:
        result["qc_reason"] = f"{type(exc).__name__}: {exc}"
    return result


def _build_nifti_qc(summary_df: pd.DataFrame, config: ConversionConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if summary_df.empty:
        return pd.DataFrame(rows)
    for _, row in summary_df[summary_df["modality"].isin(config.selected_output_order)].iterrows():
        qc = nifti_qc_one(text(row.get("nii_path", "")), config)
        rows.append({"tumor_category": text(row.get("tumor_category", "")), "subject": text(row.get("subject", "")), "modality": text(row.get("modality", "")), "record_id": text(row.get("record_id", "")), **qc})
    return pd.DataFrame(rows)


def preflight_paths(df: pd.DataFrame, config: ConversionConfig) -> dict[str, Any]:
    if df.empty:
        return {"sampled": 0, "folder_reachable": 0, "filelist_reachable": 0, "filelist_content_usable": 0}
    step = max(1, len(df) // max(1, config.preflight_sample_size))
    sample = df.iloc[::step]
    folder_ok = filelist_ok = content_ok = 0
    for _, row in sample.iterrows():
        values = row.to_dict()
        folder = _remap_path(values.get("source_folder_abs", ""), config)
        if folder and os.path.isdir(folder):
            folder_ok += 1
        filelist = _remap_path(values.get("filelist_abs", ""), config)
        if filelist and os.path.isfile(filelist):
            filelist_ok += 1
            if _filelist_paths(filelist, config):
                content_ok += 1
    total = len(sample)
    fraction = folder_ok / total if total else 0.0
    print("=" * 80)
    print("SOURCE PATH PREFLIGHT (sampled)")
    print(f"  sampled series          : {total}")
    print(f"  source_folder reachable : {folder_ok} ({fraction:.1%})")
    print(f"  filelist reachable      : {filelist_ok} ({filelist_ok / total:.1%})" if total else "  filelist reachable      : 0 (0.0%)")
    print(f"  filelist content usable : {content_ok} ({content_ok / total:.1%})" if total else "  filelist content usable : 0 (0.0%)")
    print(f"  active prefix remap     : {config.path_prefix_remap or '(none)'}")
    print("=" * 80)
    if fraction < config.preflight_min_reachable_fraction:
        raise RuntimeError(f"Source DICOM folders are largely unreachable ({fraction:.1%} < {config.preflight_min_reachable_fraction:.1%}). Fill --remap FROM=TO so the manifest paths match the current disk.")
    return {"sampled": total, "folder_reachable": folder_ok, "filelist_reachable": filelist_ok, "filelist_content_usable": content_ok, "folder_fraction": fraction}


def print_selection_waterfall(df: pd.DataFrame, config: ConversionConfig) -> None:
    df = df.copy()
    if "final_label" not in df.columns:
        df["final_label"] = df.get("model_prediction", "")
    df["final_label"] = df["final_label"].fillna("").astype(str).str.strip().str.upper()
    groups = df.groupby(["tumor_category", "subject"], sort=False, dropna=False)
    labels_by_subject = groups["final_label"].apply(set)
    required = set(config.required_modalities)
    allowed = set(config.selected_modalities)
    def passes(labels: set[str]) -> bool:
        return required.issubset(labels) and len(labels & allowed) >= config.min_modalities_per_subject
    print("=" * 80)
    print("SELECTION WATERFALL (label / study level)")
    print(f"  selected modalities     : {config.selected_output_order}")
    print(f"  require modalities      : {sorted(required)}")
    print(f"  min modalities/subject  : {config.min_modalities_per_subject}")
    print(f"  subjects                : {len(groups)}")
    for modality in config.selected_output_order:
        print(f"    has label {modality:<6}: {int(labels_by_subject.apply(lambda labels, m=modality: m in labels).sum())}")
    print(f"  label-level eligible    : {int(labels_by_subject.apply(passes).sum())}")
    same_study = 0
    for _, group in groups:
        for _, study in group.groupby("StudyInstanceUID", sort=False, dropna=False):
            if passes(set(study["final_label"])):
                same_study += 1
                break
    print(f"  same-study eligible     : {same_study}")
    print("  note: metadata rescue can recover missing modalities; acquisition-plane and original-series QC can still reduce the final count.")
    print("=" * 80)


def _read_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if "final_label" not in df.columns and "model_prediction" not in df.columns:
        raise RuntimeError("Manifest must contain final_label or model_prediction")
    for column in ("tumor_category", "subject", "StudyInstanceUID", "source_folder_abs", "filelist_abs", "n_dicoms"):
        if column not in df.columns:
            df[column] = ""
    if "final_label" not in df.columns:
        df["final_label"] = df["model_prediction"]
    return df


def run_conversion(config: ConversionConfig) -> dict[str, str]:
    df = _read_manifest(config.manifest_csv)
    preflight_paths(df, config)
    if config.diagnose_only:
        print_selection_waterfall(df, config)
        return {"diagnose": "selection waterfall printed; no conversion performed"}
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = config.output_dir / "manifest_with_orientation_cache.csv"
    if cache_path.is_file():
        enriched = pd.read_csv(cache_path, dtype=str, low_memory=False)
    else:
        enriched = enrich_manifest_with_dicom_qc(df, config)
        _atomic_dataframe(enriched, cache_path)
    print_selection_waterfall(enriched, config)
    eligible_df, selected_df, qc_df = build_selection_tables(enriched, config)
    _atomic_dataframe(eligible_df, config.output_dir / "eligible_four_modality_subjects.csv")
    _atomic_dataframe(selected_df, config.output_dir / "selected_four_modality_series.csv")
    _atomic_dataframe(qc_df, config.output_dir / "axial_four_modality_qc_subjects.csv")
    if selected_df.empty:
        summary_df = pd.DataFrame()
    else:
        groups = [(key, group.copy()) for key, group in selected_df.groupby(["tumor_category", "subject"], sort=False)]
        records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, config.workers)) as executor:
            futures = {executor.submit(_convert_subject, key, group, config): key for key, group in groups}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting subjects", unit="subject", dynamic_ncols=True):
                key = futures[future]
                try:
                    records.extend(future.result())
                except Exception as exc:
                    records.append({"tumor_category": key[0], "subject": key[1], "modality": "SUBJECT", "status": "subject_failed", "error": f"{type(exc).__name__}: {exc}"})
        summary_df = pd.DataFrame(records)
    summary_path = config.output_dir / "conversion_summary.csv"
    _atomic_dataframe(summary_df, summary_path)
    qc_output = _build_nifti_qc(summary_df, config)
    _atomic_dataframe(qc_output, config.output_dir / "nifti_qc.csv")
    atomic_json({"schema_version": "1.0", "manifest_csv": str(config.manifest_csv.resolve()), "selected_modalities": list(config.selected_modalities), "required_modalities": sorted(config.required_modalities), "min_modalities_per_subject": config.min_modalities_per_subject, "eligible_subjects": len(eligible_df), "selected_series": len(selected_df)}, config.output_dir / "conversion_config.json")
    return {"eligible": str(config.output_dir / "eligible_four_modality_subjects.csv"), "selected": str(config.output_dir / "selected_four_modality_series.csv"), "summary": str(summary_path), "nifti_qc": str(config.output_dir / "nifti_qc.csv")}
