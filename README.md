# Motion-S: Text-to-Sign Motion Generation Baseline.

## 📌 Overview

This project provides a baseline pipeline for the Motion-S Hierarchical Text-to-Motion Generation for Sign Language Kaggle competition.  
The goal is to generate realistic human motion sequences representing sign language gestures from natural language sentences.
Each training sample contains:  
1.A spoken language sentence.  
2.Sign language gloss annotations.  
3.A motion sequence represented as skeletal motion features.  
This repository implements a simple yet strong baseline that maps:
$ Sentence → Motion Sequence (T × 668)
The pipeline is designed to be modular and extendable to hierarchical modeling (Text → Gloss → Motion) and diffusion-based approaches.

## 📂 Dataset Structure

The Motion-S dataset follows a hierarchical organization where each sample is stored in its own directory containing metadata and motion files.

```
Motion-Features/
 └─ {sample_id}/
     ├ metadata.txt        # Sentence and gloss annotations
     ├ {sample_id}.bvh     # Raw skeletal motion (BVH format)
     └ {sample_id}.npy     # Preprocessed motion features (T × 668)

Train/                     # Primary training data (~12k samples)
 └─ {sample_id}/
     ├ metadata.txt        # Textual ground truth
     └ {sample_id}.bvh     # Skeleton hierarchy + motion frames

train.csv                  # Mapping of sample_id for training
test.csv                   # Sample IDs and sentences for inference
sample_submission.csv      # Submission template
```

### 📄 Metadata File (`metadata.txt`)

Each sample directory contains:

- **SENTENCE** — Natural language sentence
- **GLOSS** — Sign language gloss sequence

Example:

```
SENTENCE: I am going to school
GLOSS: I GO SCHOOL
```

### 🎞 Motion Files

- **BVH (.bvh)**
  - Raw skeletal animation
  - 55-joint hierarchy
  - 30 FPS motion sequence

- **NPY (.npy)**
  - Shape: **(T, 668)**
  - Encodes body pose, hands, face, and SMPL-X parameters
  - Used directly for model training

### 🧠 Hierarchical Learning Signal

The dataset naturally supports multi-stage modeling:

```
Text → Gloss → Motion
```

This enables research directions such as gloss-conditioned generation, motion tokenization, and diffusion-based synthesis.
