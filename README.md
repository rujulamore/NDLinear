# NdLinear Sentiment Classifier 🚀

This project demonstrates a lightweight and efficient sentiment classification model built using [NdLinear](https://github.com/ensemble-core/NdLinear), an alternative to traditional linear layers that preserves input structure and reduces parameter count. We fine-tune a DistilBERT encoder with an NdLinear classifier on the SST2 dataset from the GLUE benchmark.

---

## 🧠 Model Architecture

- **Encoder**: `distilbert-base-uncased`
- **Classifier**: Custom `NdLinear` layer (replaces standard dense head)
- **Task**: Binary sentiment classification (`positive` vs `negative`)
- **Dataset**: [GLUE/SST2](https://huggingface.co/datasets/glue)

---

## 🛠 Installation

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Run Training

```bash
python ndlinear_sentiment.py
```

> The model is trained on a 1,000-sample subset of SST2 for quick experimentation.

---

## 📊 Sample Output

```
✅ Eval Accuracy: 0.8650
```

---

## 📁 File Structure

```
.
├── ndlinear_sentiment.py     # Main training script
├── requirements.txt
├── README.md
└── results/                  # Output logs and checkpoints
```

---

## ✨ Highlights

- 🔁 Drop-in replacement of `nn.Linear` with `NdLinear`
- 🔍 Works with HuggingFace `Trainer`
- 📦 Clean integration with `transformers`, `datasets`, and `evaluate`
- 🧪 Trained using `SequenceClassifierOutput` for compatibility

---

## 🤝 Acknowledgements

- Built using [NdLinear by Ensemble AI](https://github.com/ensemble-core/NdLinear)
- Dataset: [GLUE Benchmark (SST2)](https://huggingface.co/datasets/glue)

---

## 📬 Contact

**Rujula More**  
[morer@oregonstate.edu](mailto:morer@oregonstate.edu)  
[LinkedIn](https://www.linkedin.com/in/rujula-more-19b8721a6)
