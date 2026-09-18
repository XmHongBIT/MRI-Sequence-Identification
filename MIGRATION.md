# Migration notes

This repository is the reusable version of the scripts originally kept in the parent `py/` directory.

## Kept behavior

- Original DICOM files are read-only.
- Mixed `SeriesInstanceUID` folders are split into logical series.
- A stable series identifier is generated from category, subject, source folder and Series UID.
- One representative middle slice is normalized to a padded square preview.
- All series for one subject are compared jointly by the local VLM.
- Labels may repeat and missing labels are allowed.
- Raw model responses, per-subject JSON, CSV manifests and error logs are saved.
- Conversion enforces axial acquisition, same Study selection and post-conversion NIfTI QC.

## Deliberate cleanup

- The old four recognition variants are represented by one configurable recognizer.
- Private Linux paths and model paths are command-line arguments.
- Model loading is lazy, so `--help`, conversion-only use and static checks do not load CUDA or Transformers.
- Generated data is kept outside the source tree and is ignored by Git.

## Compatibility note

The original `py/` scripts remain available as historical references. Their manifests can be used with the new `convert` command when they contain the core fields `final_label`, `source_folder_abs`, `filelist_abs`, `SeriesInstanceUID`, `StudyInstanceUID`, and `n_dicoms`.
