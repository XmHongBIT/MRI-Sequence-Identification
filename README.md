# MRI-Sequence-Identification

面向临床原始脑 MRI 的序列识别与 NIfTI 标准化流水线。项目把真实放射科输入中的逻辑 Series 重建、跨序列视觉识别、DICOM 物理参数校验、同一 Study 选择、`dcm2niix` 转换和 NIfTI QC 组织成一个可复用的 GitHub 项目。

> 本项目是研究/工程工具，不是医疗器械，也不能替代放射科医生审核。临床使用前必须完成机构级验证、隐私合规和人工 QC。

## 处理逻辑

```text
原始 DICOM
   ↓
按 SeriesInstanceUID 重建逻辑序列
   ↓
每个序列选代表性中间层，并保留 DICOM metadata
   ↓
Lingshu/Qwen2.5-VL 同时比较同一 subject 的所有序列
   ↓
输出 T1 / T1CE / T2 / FLAIR / DWI / ADC / SWI 等标签
   ↓
轴位筛选 + FLAIR 物理参数校验 + 同 Study 约束
   ↓
dcm2niix → NIfTI
   ↓
shape / spacing / affine / file-size / slice-count QC
```

设计原则：序列标签允许重复；某些序列可以缺失；原始 DICOM 不移动、不重命名、不修改；每个逻辑序列使用稳定 `record_id`；每个 subject 以 JSON 作为断点续跑的 canonical record；所有中间产物都能回溯到原始 DICOM。

## 输入目录

```text
dicom_root/
├── tumor_category_a/
│   ├── subject_001/
│   │   ├── series_folder_1/
│   │   │   ├── 0001.dcm
│   │   │   └── ...
│   │   └── series_folder_2/
│   └── subject_002/
└── tumor_category_b/
```

如果一个文件夹混入多个 `SeriesInstanceUID`，程序会拆成多个逻辑序列，而不会把它们静默合并。

## 安装

先安装与你的 NVIDIA 驱动匹配的 PyTorch，再安装项目：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[vlm]"
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[vlm]"
```

同时需要系统命令 `dcm2niix`：

```bash
dcm2niix -h
```

模型默认按本地目录加载。若确实要让 Transformers 下载模型，显式加入 `--allow-model-download`。

## 运行

### 1. 只识别序列

```bash
python -m mri_sequence_identification recognize \
  --dicom-root /data/raw_dicom \
  --output-dir /data/sequence_results \
  --model-path /models/lingshu_32b \
  --max-subjects-per-category 400
```

识别结果主要包括：

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

### 2. 选择四模态并转换 NIfTI

```bash
python -m mri_sequence_identification convert \
  --manifest-csv /data/sequence_results/manifests/dcm2nii_manifest.csv \
  --output-dir /data/nifti_output \
  --dcm2niix /usr/local/bin/dcm2niix
```

转换默认只接受同一 Study 的轴位 `T1/T1CE/T2/FLAIR`。FLAIR 默认要求名称线索或合理的 `TI/TE/TR`；需要宽松结果时显式加 `--non-strict-flair`。

### 3. 一键运行

```bash
python -m mri_sequence_identification pipeline \
  --dicom-root /data/raw_dicom \
  --output-dir /data/sequence_results \
  --nifti-output-dir /data/nifti_output \
  --model-path /models/lingshu_32b
```

### 4. 单个 subject 试跑

```bash
python -m mri_sequence_identification recognize \
  --dicom-root /data/raw_dicom \
  --output-dir /data/debug_results \
  --model-path /models/lingshu_32b \
  --target-category glioma \
  --target-subject subject_001
```

## 输出与人工复核

转换阶段会生成：

```text
eligible_four_modality_subjects.csv
selected_four_modality_series.csv
conversion_summary.csv
nifti_qc.csv
conversion_config.json
```

建议在大规模运行前先抽查 `selected_four_modality_series.csv` 中的 `SeriesDescription`、`ProtocolName`、`TR/TE/TI`、`selection_source`，再检查 `nifti_qc.csv`。低置信度、短序列、解析失败、混合 UID 文件夹会标记为 `review_required`。

## 从原始脚本迁移

| 原脚本 | 新入口 |
| --- | --- |
| `lingshu_dicom_sequence_read*.py` | `recognize` |
| `dcm2nii_four_modalities_fast_safe.py` | `convert` |

历史脚本中的服务器路径、输出目录、GPU 数量和数据集上限不再写死在源码里，均可通过命令行参数调整。原始 `py/` 目录未被修改。

## 数据安全与复现

不要把真实 DICOM、NIfTI、患者姓名、住院号、模型权重或生成的 CSV/JSON 提交到 GitHub。仓库的 `.gitignore` 已默认忽略这些文件，但提交前仍应人工检查：

```bash
git status --short
git diff --stat
```

建议公开仓库只包含代码、文档和脱敏的小型测试数据；模型权重使用其原始许可证单独管理。
