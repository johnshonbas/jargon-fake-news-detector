# Jargon Fake News Detector 🕵️‍♂️

A high-performance **BERT-GRU Hybrid** model designed to classify news headlines as Real or Fake with high confidence.

### 📥 Download Model Weights
Because the model file is 420MB, it is hosted on Hugging Face:
👉 **[Download .pth Weights Here](https://huggingface.co/hemanthcharan/jargon-fake-news-detector/tree/main)**

### 📊 Model Specifications
- **Backbone:** BERT-base-uncased
- **Layers:** Bidirectional GRU + Custom Attention
- **Total Parameters:** 110,205,186 (Fully Trainable)
- **Status:** ✅ Verified Healthy (No NaNs, stable weight distribution)

### 🛠️ Quick Start
1. Clone this repository.
2. Install requirements:
   ```bash
   pip install torch transformers
