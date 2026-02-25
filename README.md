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
