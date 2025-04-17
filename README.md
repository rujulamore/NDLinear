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
