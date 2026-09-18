from __future__ import annotations

import gc
import json
import os
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .dicom import LogicalSeries, collect_subject_series, dicom_to_pil
from .utils import atomic_csv, atomic_json, natural_key, safe_name


VALID_LABELS = {"T1", "T1CE", "T2", "FLAIR", "DWI", "ADC", "SWI", "LOCALIZER", "OTHER"}
VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}
CONVERTIBLE_LABELS = {"T1", "T1CE", "T2", "FLAIR", "DWI", "ADC", "SWI"}
PRIMARY_TARGET_LABELS = {"T1", "T1CE", "T2", "FLAIR"}

PROMPT = """
You are an expert neuroradiologist identifying brain MRI sequences.
The images below are the representative middle slice of different logical DICOM series from one patient.
Compare all series jointly before labeling each one. Labels may repeat and some sequences may be absent.

Allowed labels: T1, T1CE, T2, FLAIR, DWI, ADC, SWI, LOCALIZER, OTHER.

Use these distinctions:
- T1 vs T1CE: compare T1-like series directly; stronger lesion, vascular, meningeal or choroid-plexus enhancement favors T1CE.
- T2 vs FLAIR: T2 usually has bright CSF; FLAIR has T2-like lesion/edema signal with suppressed dark CSF.
- DWI/ADC and SWI require characteristic diffusion-map or susceptibility/blooming appearance.
- LOCALIZER is a scout/planning acquisition. OTHER includes derived, secondary, parameter-map, screenshot, or uncertain images.
- DICOM metadata is auxiliary evidence only; never classify from folder name, series number, or slice count alone.

Return exactly one line per input series, in the same order:
SERIES_001 | LABEL | CONFIDENCE | concise comparison-based reason
No markdown, headings, bullets, or extra commentary.
""".strip()


@dataclass
class RecognizerConfig:
    dicom_root: Path
    output_dir: Path
    model_path: str
    image_size: int = 512
    max_series_per_subject: int = 32
    max_subjects_per_category: int | None = 400
    target_category: str | None = None
    target_subject: str | None = None
    resume: bool = True
    save_previews: bool = True
    runtime_reserve_gib: float = 5.0
    cpu_offload_gib: int = 96
    max_new_tokens: int = 896
    local_files_only: bool = True


def _load_vlm(config: RecognizerConfig) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("VLM dependencies are missing. Install with `pip install -e '.[vlm]'`.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; the VLM recognizer requires a CUDA device.")
    processor = AutoProcessor.from_pretrained(config.model_path, trust_remote_code=True, local_files_only=config.local_files_only)
    max_memory: dict[int | str, str] = {}
    for gpu_id in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        max_memory[gpu_id] = f"{max(8, int(total_gib - config.runtime_reserve_gib))}GiB"
    max_memory["cpu"] = f"{config.cpu_offload_gib}GiB"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_path, torch_dtype=torch.bfloat16, device_map="balanced", max_memory=max_memory,
        low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=config.local_files_only,
    )
    model.eval()
    return processor, model, torch


def _model_input_device(model: Any, torch: Any) -> Any:
    device_map = getattr(model, "hf_device_map", {})
    for keyword in ("visual", "vision", "embed_tokens", "language_model"):
        for module, device in device_map.items():
            if keyword in str(module).lower():
                return torch.device(device if isinstance(device, str) else f"cuda:{device}")
    return next(model.parameters()).device


def _build_messages(series: list[LogicalSeries], image_size: int) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": PROMPT}]
    for index, item in enumerate(series, start=1):
        metadata = json.dumps(item.metadata, ensure_ascii=False)
        content.append({"type": "text", "text": f"\nSERIES_{index:03d} metadata (auxiliary): {metadata}"})
        content.append({"type": "image", "image": dicom_to_pil(item.middle_dicom_abs, image_size)})
    return [{"role": "user", "content": content}]


def _classify(processor: Any, model: Any, torch: Any, series: list[LogicalSeries], config: RecognizerConfig) -> str:
    from qwen_vl_utils import process_vision_info
    messages = _build_messages(series, config.image_size)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(_model_input_device(model, torch))
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=config.max_new_tokens, do_sample=False, use_cache=True)
    new_tokens = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def parse_response(response: str, series: list[LogicalSeries]) -> list[dict[str, Any]]:
    pattern = re.compile(r"SERIES[_\s-]*(\d+)\s*\|\s*([A-Za-z0-9_+-]+)\s*\|\s*([A-Za-z]+)\s*\|?\s*(.*)", re.I)
    predictions: dict[int, tuple[str, str, str]] = {}
    for match in pattern.finditer(response):
        index = int(match.group(1))
        label = match.group(2).upper().replace("-", "")
        confidence = match.group(3).upper()
        reason = match.group(4).strip()
        if label not in VALID_LABELS:
            label = "OTHER"
        if confidence not in VALID_CONFIDENCE:
            confidence = "LOW"
        predictions[index] = label, confidence, reason
    results: list[dict[str, Any]] = []
    for index, item in enumerate(series, start=1):
        label, confidence, reason = predictions.get(index, ("OTHER", "LOW", "No valid structured result returned by model."))
        folder_uid_clean = item.folder_uid_group_count == 1 and item.series_uid != "__MISSING_SERIES_UID__"
        structured_ok = index in predictions
        row = item.as_dict()
        row.update({
            "model_series_id": f"SERIES_{index:03d}", "model_prediction": label,
            "model_confidence": confidence, "model_reason": reason, "final_label": label,
            "structured_parse_ok": structured_ok, "is_primary_target": label in PRIMARY_TARGET_LABELS,
            "is_convertible_label": label in CONVERTIBLE_LABELS, "source_folder_is_uid_clean": folder_uid_clean,
            "dcm2nii_ready_direct": folder_uid_clean and label in CONVERTIBLE_LABELS and confidence != "LOW" and structured_ok,
            "review_required": confidence == "LOW" or label in {"OTHER", "LOCALIZER"} or item.short_series or not structured_ok or not folder_uid_clean,
            "manual_label": "", "manual_qc_status": "", "manual_note": "",
            "recommended_output_stem": f"{safe_name(item.subject)}__{item.record_id}__{label}",
            "recommended_output_dir_rel": os.path.join(safe_name(item.tumor_category), safe_name(item.subject), label),
        })
        results.append(row)
    return results


def _subject_entries(config: RecognizerConfig) -> list[dict[str, str]]:
    if not config.dicom_root.is_dir():
        raise FileNotFoundError(f"DICOM root does not exist: {config.dicom_root}")
    entries: list[dict[str, str]] = []
    categories = sorted((path for path in config.dicom_root.iterdir() if path.is_dir() and not path.name.startswith("_")), key=lambda path: natural_key(path.name))
    for category in categories:
        if config.target_category and category.name != config.target_category:
            continue
        subjects = sorted((path for path in category.iterdir() if path.is_dir()), key=lambda path: natural_key(path.name))
        if config.target_subject:
            subjects = [path for path in subjects if path.name == config.target_subject]
        elif config.max_subjects_per_category is not None:
            # Keep completed subjects in the cohort on restart. This makes a cap
            # safe for long-running studies when new cases are added later.
            completed = [
                path for path in subjects
                if (config.output_dir / "subjects" / safe_name(category.name) / safe_name(path.name) / "_COMPLETE").is_file()
            ]
            unfinished = [path for path in subjects if path not in completed]
            subjects = completed + unfinished[: max(0, config.max_subjects_per_category - len(completed))]
        entries.extend({"tumor_category": category.name, "subject": path.name, "subject_dir": str(path)} for path in subjects)
    return entries


def run_recognition(config: RecognizerConfig) -> dict[str, str]:
    output = config.output_dir
    dirs = {name: output / folder for name, folder in {
        "subjects": "subjects", "previews": "middle_slice_preview", "responses": "raw_response",
        "filelists": "series_filelists", "manifests": "manifests", "logs": "logs",
    }.items()}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    entries = _subject_entries(config)
    processor, model, torch = _load_vlm(config)
    all_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    resumed = 0
    for entry in tqdm(entries, desc="Recognizing subjects", unit="subject"):
        category, subject, subject_dir = entry["tumor_category"], entry["subject"], Path(entry["subject_dir"])
        result_dir = dirs["subjects"] / safe_name(category) / safe_name(subject)
        result_json, marker = result_dir / "series_classification.json", result_dir / "_COMPLETE"
        if config.resume and marker.exists() and result_json.exists():
            try:
                all_rows.extend(json.loads(result_json.read_text(encoding="utf-8"))["series"])
                resumed += 1
            except Exception as exc:
                errors.append({"subject": subject, "error_type": "INVALID_RESUME_RESULT", "error_message": str(exc)})
            continue
        try:
            series = collect_subject_series(subject_dir, config.dicom_root, category, subject, dirs["filelists"])
            if not series:
                raise RuntimeError("No valid image DICOM series found.")
            if len(series) > config.max_series_per_subject:
                raise RuntimeError(f"{len(series)} logical series exceeds max_series_per_subject={config.max_series_per_subject}")
            if config.save_previews:
                preview_dir = dirs["previews"] / safe_name(category) / safe_name(subject)
                preview_dir.mkdir(parents=True, exist_ok=True)
                for index, item in enumerate(series, start=1):
                    dicom_to_pil(item.middle_dicom_abs, config.image_size).save(preview_dir / f"SERIES_{index:03d}__{item.record_id}.png")
            response = _classify(processor, model, torch, series, config)
            rows = parse_response(response, series)
            payload = {"schema_version": "1.0", "created_at": datetime.now().isoformat(timespec="seconds"), "model_path": config.model_path, "dicom_root": str(config.dicom_root.resolve()), "tumor_category": category, "subject": subject, "subject_dir_abs": str(subject_dir.resolve()), "number_of_logical_series": len(rows), "raw_model_response": response, "series": rows}
            atomic_json(payload, result_json)
            marker.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
            response_dir = dirs["responses"] / safe_name(category)
            response_dir.mkdir(parents=True, exist_ok=True)
            (response_dir / f"{safe_name(subject)}.txt").write_text(response, encoding="utf-8")
            all_rows.extend(rows)
        except Exception as exc:
            errors.append({"time": datetime.now().isoformat(timespec="seconds"), "tumor_category": category, "subject": subject, "subject_dir_abs": str(subject_dir.resolve()), "error_type": type(exc).__name__, "error_message": str(exc)})
            traceback.print_exc()
        finally:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    all_rows.sort(key=lambda row: (natural_key(row.get("tumor_category", "")), natural_key(row.get("subject", "")), natural_key(row.get("series_folder", ""))))
    manifest_fields = list(dict.fromkeys(key for row in all_rows for key in row))
    atomic_csv(all_rows, dirs["manifests"] / "series_manifest.csv", manifest_fields)
    dcm2nii_fields = ["record_id", "tumor_category", "subject", "final_label", "model_prediction", "model_confidence", "review_required", "dcm2nii_ready_direct", "source_folder_is_uid_clean", "source_folder_abs", "source_folder_rel", "SeriesInstanceUID", "StudyInstanceUID", "n_dicoms", "filelist_abs", "recommended_output_dir_rel", "recommended_output_stem", "SeriesDescription", "ProtocolName", "ImageType", "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle", "ContrastBolusAgent"]
    atomic_csv([{field: row.get(field, "") for field in dcm2nii_fields} for row in all_rows], dirs["manifests"] / "dcm2nii_manifest.csv", dcm2nii_fields)
    atomic_csv(errors, dirs["manifests"] / "processing_errors.csv")
    atomic_json({"schema_version": "1.0", "created_at": datetime.now().isoformat(timespec="seconds"), "config": {key: str(value) for key, value in config.__dict__.items()}, "subjects_total": len(entries), "subjects_resumed": resumed, "series_total": len(all_rows)}, output / "run_config.json")
    return {"series_manifest": str(dirs["manifests"] / "series_manifest.csv"), "dcm2nii_manifest": str(dirs["manifests"] / "dcm2nii_manifest.csv"), "processing_errors": str(dirs["manifests"] / "processing_errors.csv")}
