# Standardizing Longitudinal Chest X-ray Report Evaluation via Large Language Model Annotation

This repository contains the official implementation of the paper:

**“Standardizing Longitudinal Chest X-ray Report Evaluation via Large Language Model Annotation.”**

The paper has been accepted for publication in Expert Systems with Applications. GitHub will be ready soon.

## Abstract

Longitudinal information in radiology reports refers to the sequential tracking of findings across multiple imaging-based examinations over time, which is crucial for monitoring disease progression and guiding clinical decisions. Many recent automated radiology report generation methods are designed to capture longitudinal information; however, validating their performance is challenging. There is no proper tool to consistently label temporal changes in both ground-truth and model-generated texts for meaningful comparisons. Large language models (LLMs) offer a promising annotation, as they are capable of capturing nuanced linguistic patterns and semantic similarities without extensive manual intervention. In this study, we therefore propose an LLM-based pipeline to automatically annotate longitudinal information in radiology reports. The pipeline first identifies sentences containing relevant information and then extracts disease progression information expressed in text. We evaluate and compare five mainstream LLMs on these two tasks using 500 manually annotated reports. Considering both efficiency and performance, Qwen2.5-32B was subsequently selected and used to annotate another 95,169 reports from the public MIMIC-CXR dataset. Our Qwen2.5-32B-annotated dataset provided us with a standardized benchmark for evaluating report generation models. Using this new benchmark, we assessed seven state-of-the-art report generation models on their ability to producing longitudinal information. Our LLM-based annotation methods outperform existing annotation solutions, improving F1 scores from 82.7\% to 94.0\% (+11.3\%) for longitudinal information detection and from 78.5\% to 83.8\% (+5.3\%) for disease progression information extraction. The LLMs show relatively stable gains in the no change category, but performs less effectively on the worsened category. In conclusion, this work demonstrates the potential of LLMs for efficient and effective medical report annotation and establishes a standardized evaluation framework for longitudinal chest X-ray report generation. 

## 📄 Paper Link:

## 🔗 Dataset Access:

## 💻 Code and Evaluation

### 1. Evaluate Your Report Generation Model with Our Tool

### 2. Reproducing the Results in Our paper

#### 2.1 Comparison Our Pipeline with Traditional Methods

##### Performance of LLMs on Longitudinal Annotation

For all large language models, the decoding method is greedy decoding.

Model | Model Size | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|
| ImaGenome silver | -- | 90.2 | 98.5 | 71.2 | 82.7 |
| MedGemma | 27B | 95.5 (5.3 ↑) | 92.7 (5.8 ↓) | 93.7 (22.5 ↑) | 93.2 (10.5 ↑) |
| MedResearcher-R1 | 32B | 94.0 (3.8 ↑) | 95.4 (3.1 ↓) | 86.0 (14.8 ↑) | 90.5 (7.8 ↑) |
| Qwen2.5 | 32B | 94.8 (4.6 ↑) | 95.1 (3.4 ↓) | 88.8 (17.6 ↑) | 91.8 (9.1 ↑) |
| Llama3.3 | 70B | 95.9 (5.7 ↑) | 91.4 (7.1 ↓) | 96.8 (25.6 ↑) | 94.0 (11.3 ↑) |
| Qwen2.5 | 72B | 93.4 (3.2 ↑) | 94.4 (4.1 ↓) | 85.1 (13.9 ↑) | 89.5 (6.8 ↑) |

##### Performance of LLMs on Disease Progression Annotation

The micro-average metric is computed over the three classes: no change, improved, and worsened. For all large language models, the decoding method is greedy decoding. ImaG. silver and MedR-R1 refer to ImaGenome silver and MedResearcher-R1, respectively. `*` indicates statistically significant improvement over ImaGenome Silver (95% CI of the difference does not contain zero).

<table> <thead> <tr> <th><small>Model</small></th> <th><small>Model Size</small></th> <th><small>Micro Acc.</small></th> <th><small>Micro F1</small></th> <th><small>Macro F1</small></th> <th><small>No Change F1</small></th> <th><small>Improved F1</small></th> <th><small>Worsened F1</small></th> </tr> </thead> <tbody> <tr> <td><small>ImaG. silver</small></td> <td><small>--</small></td> <td><small>88.7<br>[87.3, 90.0]</small></td> <td><small>78.5<br>[75.7, 81.1]</small></td> <td><small>77.1<br>[73.8, 80.1]</small></td> <td><small>82.1<br>[78.8, 85.0]</small></td> <td><small>76.6<br>[69.1, 82.5]</small></td> <td><small>72.5<br>[66.7, 77.8]</small></td> </tr> <tr> <td><small>MedGemma</small></td> <td><small>27B</small></td> <td><small><strong>91.4*</strong> (2.7 ↑)<br>[90.2, 92.6]</small></td> <td><small><strong>83.7*</strong> (5.2 ↑)<br>[81.4, 86.2]</small></td> <td><small><strong>81.2*</strong> (4.1 ↑)<br>[78.3, 84.0]</small></td> <td><small><strong>87.9*</strong> (5.8 ↑)<br>[85.4, 90.4]</small></td> <td><small>76.0 (0.6 ↓)<br>[68.5, 82.4]</small></td> <td><small><strong>79.6*</strong> (7.1 ↑)<br>[74.9, 83.8]</small></td> </tr> <tr> <td><small>MedR-R1</small></td> <td><small>32B</small></td> <td><small>90.1 (1.4 ↑)<br>[88.8, 91.5]</small></td> <td><small>79.5 (1.0 ↑)<br>[76.7, 82.4]</small></td> <td><small>76.6 (0.5 ↓)<br>[73.0, 80.0]</small></td> <td><small>84.6 (2.5 ↑)<br>[81.8, 87.6]</small></td> <td><small>74.9 (1.7 ↓)<br>[67.6, 81.3]</small></td> <td><small>70.4 (2.1 ↓)<br>[64.4, 75.9]</small></td> </tr> <tr> <td><small>Qwen2.5</small></td> <td><small>32B</small></td> <td><small><strong>91.9*</strong> (3.2 ↑)<br>[90.6, 93.1]</small></td> <td><small><strong>83.2*</strong> (4.7 ↑)<br>[80.5, 85.8]</small></td> <td><small><strong>82.1*</strong> (5.0 ↑)<br>[79.2, 85.0]</small></td> <td><small><strong>87.0*</strong> (4.9 ↑)<br>[84.4, 89.7]</small></td> <td><small><strong>87.1*</strong> (10.5 ↑)<br>[82.1, 91.6]</small></td> <td><small>72.2 (0.3 ↓)<br>[66.5, 77.6]</small></td> </tr> <tr> <td><small>Llama3.3</small></td> <td><small>70B</small></td> <td><small><strong>90.5*</strong> (1.8 ↑)<br>[89.2, 91.8]</small></td> <td><small>78.9 (0.4 ↑)<br>[75.9, 81.7]</small></td> <td><small>78.9 (1.8 ↑)<br>[75.9, 81.7]</small></td> <td><small>79.8 (2.3 ↓)<br>[76.1, 83.2]</small></td> <td><small>79.4 (2.8 ↑)<br>[73.2, 84.3]</small></td> <td><small>77.4 (4.9 ↑)<br>[73.0, 81.2]</small></td> </tr> <tr> <td><small>Qwen2.5</small></td> <td><small>72B</small></td> <td><small><strong>91.2*</strong> (2.5 ↑)<br>[89.9, 92.6]</small></td> <td><small><strong>83.8*</strong> (5.3 ↑)<br>[81.2, 86.2]</small></td> <td><small><strong>82.5*</strong> (5.4 ↑)<br>[79.2, 85.3]</small></td> <td><small><strong>87.7*</strong> (5.6 ↑)<br>[85.0, 90.1]</small></td> <td><small>83.7 (7.1 ↑)<br>[77.2, 88.7]</small></td> <td><small>76.2 (3.7 ↑)<br>[70.9, 81.1]</small></td> </tr> </tbody> </table>

#### 2.2 Evaluation of Report Generation Models


## 📚 Citation

If you use this code for your research, please cite our paper:

```bibtex

}
