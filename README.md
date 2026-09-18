# MRI-Sequence-Identification

A reusable pipeline for clinical brain MRI sequence identification and NIfTI standardization. The project organizes logical DICOM series, performs cross-series visual identification, validates DICOM acquisition metadata, selects consistent same-Study modalities, converts selected series with `dcm2niix`, and performs post-conversion NIfTI quality control.

## Workflow

```text
Raw DICOM
   ↓
Reconstruct logical series by SeriesInstanceUID
   ↓
Select a representative middle slice and preserve DICOM metadata
   ↓
Compare all series from the same subject with VLM
   ↓
Predict T1 / T1CE / T2 / FLAIR / DWI / ADC / SWI and other labels
   ↓
Axial-plane filtering + FLAIR physical validation + same-Study constraint
   ↓
dcm2niix → NIfTI
   ↓
Shape / spacing / affine / file-size / slice-count QC
```

Recommended VLM: [Lingshu 32B](https://huggingface.co/lingshu-medical-mllm/Lingshu-32B)

Design principles:

- Sequence labels may repeat; a sequence may be absent.
- Original DICOM files are never moved, renamed, or modified.
- Every logical series receives a stable `record_id`.
- Each subject has a canonical JSON record for safe resume and auditability.
- Intermediate outputs remain traceable to the original DICOM files.

## Input layout

The default input layout is:

```text
dicom_root/
├── SERIZE_a/
│   ├── subject_001/
│   │   ├── series_folder_1/
│   │   │   ├── 0001.dcm
│   │   │   └── ...
│   │   └── series_folder_2/
│   └── subject_002/
└── SERIZE_b/
```

If one source folder contains multiple `SeriesInstanceUID` values, the program splits it into multiple logical series instead of silently merging them.

## Installation

Install a PyTorch build compatible with your NVIDIA driver first, then install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[vlm]"
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[vlm]"
```

The system command `dcm2niix` is also required. Verify that it is available on your `PATH`:

```bash
dcm2niix -h
```

The model is loaded from a local directory by default. To allow Transformers to download model files, explicitly add `--allow-model-download`.

## Usage

### 1. Identify MRI sequences

```bash
python -m mri_sequence_identification recognize \
  --dicom-root /data/raw_dicom \
  --output-dir /data/sequence_results \
  --model-path /models/lingshu_32b \
```

The main recognition outputs are:

```text
sequence_results/
├── manifests/series_manifest.csv
├── manifests/dcm2nii_manifest.csv
├── manifests/processing_errors.csv
├── subjects/<category>/<subject>/series_classification.json
├── subjects/<category>/<subject>/_COMPLETE
├── middle_slice_preview/
└── series_filelists/
```

### 2. Select four modalities and convert to NIfTI

```bash
python -m mri_sequence_identification convert \
  --manifest-csv /data/sequence_results/manifests/dcm2nii_manifest.csv \
  --output-dir /data/nifti_output \
  --dcm2niix /usr/local/bin/dcm2niix
```

By default, conversion accepts only axial `T1/T1CE/T2/FLAIR` series from the same Study. FLAIR requires either a sequence-name clue or plausible `TI/TE/TR` values. Add `--non-strict-flair` only when a more permissive selection is needed.

### 3. Run the complete pipeline

```bash
python -m mri_sequence_identification pipeline \
  --dicom-root /data/raw_dicom \
  --output-dir /data/sequence_results \
  --nifti-output-dir /data/nifti_output \
  --model-path /models/lingshu_32b
```

### 4. Run a single category or subject during development

```bash
python -m mri_sequence_identification recognize \
  --dicom-root /data/raw_dicom \
  --output-dir /data/debug_results \
  --model-path /models/lingshu_32b \
  --target-category SERIZE_a \
  --target-subject subject_001
```

## Outputs and manual review

The conversion stage produces:

```text
eligible_four_modality_subjects.csv
selected_four_modality_series.csv
conversion_summary.csv
nifti_qc.csv
conversion_config.json
```

Before a large-scale run, inspect `selected_four_modality_series.csv`, especially `SeriesDescription`, `ProtocolName`, `TR/TE/TI`, and `selection_source`. Then review `nifti_qc.csv`. Low-confidence predictions, short series, parsing failures, and mixed-UID source folders are marked with `review_required`.
