# Standardizing Longitudinal Chest X-ray Report Evaluation via Large Language Model Annotation

This repository contains the official implementation of the paper:

**“Standardizing Longitudinal Chest X-ray Report Evaluation via Large Language Model Annotation.”**

The paper has been accepted for publication in Expert Systems with Applications. GitHub will be ready soon.

## Abstract

Longitudinal information in radiology reports refers to the sequential tracking of findings across multiple imaging-based examinations over time, which is crucial for monitoring disease progression and guiding clinical decisions. Many recent automated radiology report generation methods are designed to capture longitudinal information; however, validating their performance is challenging. There is no proper tool to consistently label temporal changes in both ground-truth and model-generated texts for meaningful comparisons. Large language models (LLMs) offer a promising annotation, as they are capable of capturing nuanced linguistic patterns and semantic similarities without extensive manual intervention. In this study, we therefore propose an LLM-based pipeline to automatically annotate longitudinal information in radiology reports. The pipeline first identifies sentences containing relevant information and then extracts disease progression information expressed in text. We evaluate and compare five mainstream LLMs on these two tasks using 500 manually annotated reports. Considering both efficiency and performance, Qwen2.5-32B was subsequently selected and used to annotate another 95,169 reports from the public MIMIC-CXR dataset. Our Qwen2.5-32B-annotated dataset provided us with a standardized benchmark for evaluating report generation models. Using this new benchmark, we assessed seven state-of-the-art report generation models on their ability to producing longitudinal information. Our LLM-based annotation methods outperform existing annotation solutions, improving F1 scores from 82.7\% to 94.0\% (+11.3\%) for longitudinal information detection and from 78.5\% to 83.8\% (+5.3\%) for disease progression information extraction. The LLMs show relatively stable gains in the no change category, but performs less effectively on the worsened category. In conclusion, this work demonstrates the potential of LLMs for efficient and effective medical report annotation and establishes a standardized evaluation framework for longitudinal chest X-ray report generation. 

## 📄 Paper Link:

## 🔗 Dataset Access:

## 💻 Code and Evaluation

## 1. Evaluate Your Report Generation Model with Our Tool

## 2. Reproducing the Results in Our paper

### 2.1 Comparison Our Pipeline with Traditional Methods

### 2.2 Evaluation of Report Generation Models


## 📚 Citation

If you use this code for your research, please cite our paper:

```bibtex

}
