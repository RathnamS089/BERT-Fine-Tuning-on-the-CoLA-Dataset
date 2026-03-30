
# 📖 BERT Fine-Tuning for Linguistic Acceptability (CoLA)

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Tested-EE4C2C.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9A814.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A complete, end-to-end PyTorch pipeline for fine-tuning a pre-trained BERT model (`bert-base-uncased`) on the **Corpus of Linguistic Acceptability (CoLA)** dataset. This model is trained to perform binary classification, determining whether a given English sentence is grammatically acceptable or unacceptable.

---

## 📑 Table of Contents
- [Features](#-features)
- [Dataset](#-dataset)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Expected Output](#-expected-output)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Features
* **Cross-Platform Hardware Acceleration:** Automatically detects and utilizes NVIDIA GPUs (CUDA), Apple Silicon (MPS), or falls back to standard CPU.
* **Automated Data Pipeline:** Downloads, extracts, and parses the CoLA dataset directly from the source using `wget`. No manual data wrangling required.
* **Robust Tokenization:** Utilizes Hugging Face's `BertTokenizer` to accurately encode text, apply attention masks, and handle padding/truncation (Max Length: 64).
* **Complete Training & Validation Loop:** Implements standard PyTorch training loops with the `AdamW` optimizer, learning rate scheduling with warmup, and gradient clipping to prevent exploding gradients.
* **Formatted Statistics:** Outputs a clean, easy-to-read Pandas DataFrame summarizing training loss, validation loss, accuracy, and elapsed time per epoch.

---

## 📊 Dataset
The **Corpus of Linguistic Acceptability (CoLA)** consists of over 10,000 English sentences sourced from linguistics literature. Each sentence is annotated with a binary label indicating whether it is grammatically acceptable (`1`) or unacceptable (`0`).

* **Source:** [NYU CoLA Dataset](https://nyu-mll.github.io/CoLA/)
* The script automatically downloads the `cola_public_1.1.zip` dataset during its first run.

---

## 🛠 Prerequisites
Ensure you have **Python 3.7 or higher** installed. The project relies heavily on PyTorch and the Hugging Face ecosystem.

Key dependencies:
* `torch`
* `transformers`
* `pandas`
* `numpy`
* `wget`

---

## 💻 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/bert-cola-finetuning.git](https://github.com/yourusername/bert-cola-finetuning.git)
   cd bert-cola-finetuning
