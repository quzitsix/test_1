---
configs:
  - config_name: qa_metadata
    data_files:
      - split: train
        path: egomonth_qa_metadata.csv
  - config_name: sample_video_manifest
    data_files:
      - split: train
        path: sample_video_manifest.csv
license: other
license_name: egomonth-research-only-data-use-agreement
license_link: https://huggingface.co/datasets/anonymous-egomonth/egomonth-dataset/resolve/main/LICENSE_RESEARCH_ONLY.txt
language: en
task_categories:
  - visual-question-answering
  - video-text-to-text
tags:
  - egocentric
  - video
  - multimodal
  - reasoning
size_categories:
  - 1K<n<10K
---
# EgoMonth Dataset

## Overview

EgoMonth is a month-level egocentric video question-answering benchmark for evaluating long-term spatiotemporal memory in multimodal large language models. The dataset focuses on daily-life first-person videos and QA tasks that require temporal indexing, spatial grounding, multi-video reasoning, and long-horizon memory.

This repository provides QA metadata, structured annotations, representative anonymized sample videos, and baseline evaluation scripts. Full raw videos follow a controlled research-access process because month-level first-person recordings may contain privacy-sensitive information.

## Tasks

- Visual question answering
- Temporal reasoning
- Spatial reasoning
- Long-term memory and multi-evidence reasoning

## Repository Contents

- annotations/:
  Structured JSON annotations.
  - global_json_list/: full annotation files for the benchmark, including QA, event, object, and place annotations.
  - sample_json_list/: annotations corresponding to the publicly hosted sample-video subset.

- videos/:
  Representative anonymized sample videos for quick inspection, validation, and reviewer-facing benchmarking.

- egomonth_qa_metadata.csv:
  A flattened QA metadata table generated from global annotations. This is the default lightweight table for browsing the benchmark.

- sample_video_manifest.csv:
  Metadata for the hosted sample videos, including file paths, participant/sample IDs, file sizes, annotation paths, anonymization status, and access notes.

- scripts/:
  Baseline inference and evaluation scripts for reproducing the multiple-choice benchmark protocol.

## Evaluation Scripts

The baseline scripts support batch evaluation through `DATASET_LIST`. Each dataset ID maps to:

```text
annotations/global_json_list/<dataset_id>/QA.json
```

Results and logs are written to `results/` and `logs/` by default. Before running, check the configuration block at the top of each script and set model paths, video roots, output directories, and hardware/API options as needed.

For Qwen-based scripts, videos are loaded from the local `videos/` root. For the Gemini script, upload the referenced videos to Google Cloud Storage and update `video_path` fields in the QA JSON files to GCS URIs before inference.

## Access

EgoMonth is distributed under the EgoMonth Research-Only Data Use Agreement, not under a Creative Commons license. By accessing or using the dataset files, users agree to the research-only terms in `LICENSE_RESEARCH_ONLY.txt`.

Permitted use is limited to academic and non-commercial research, including benchmark evaluation, diagnostic analysis, and reproducibility studies. Users may not redistribute, re-host, mirror, sell, sublicense, or otherwise share the dataset files or derived copies. Users may not use the dataset for surveillance, profiling, biometric identification, sensitive-attribute inference, commercial decision-making, or attempts to identify participants.

The public repository contains metadata, annotations, scripts, and a curated anonymized sample-video subset. Full raw videos are hosted externally and require additional controlled research-access approval.

## Ethics

The dataset is intended for research use only. Videos were collected with participant consent, and the public sample videos are anonymized to reduce privacy risk. Users must not attempt to identify participants, recover sensitive information, contact recorded individuals, or use the data for surveillance, profiling, biometric identification, sensitive-attribute inference, or commercial decision-making.

## Bias & Limitations

EgoMonth reflects the environments, routines, recording devices, and participant demographics present in the collected data. As a first-person video benchmark, it may include partial observability, occlusions, motion blur, and viewpoint-specific bias.

The benchmark is designed for evaluating long-horizon egocentric video understanding. It should not be treated as a demographically complete representation of daily life or as evidence for sensitive personal, clinical, legal, or biometric inference.
