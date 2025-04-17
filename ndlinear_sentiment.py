import torch
from torch import nn
from ndlinear import NdLinear
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset
import evaluate
import numpy as np
from transformers import Trainer
from transformers.training_args import TrainingArguments



dataset = load_dataset("glue", "sst2")


for split in dataset:
    print(f"- {split}: {len(dataset[split])} examples")


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# Tokenization function
def tokenize(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

tokenized = dataset.map(tokenize, batched=True)

tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])


class NdLinearSentimentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        hidden_size = self.encoder.config.hidden_size  # typically 768

        # NdLinear expects multi-dimensional inputs, so we reshape before passing
        self.ndlinear = NdLinear(input_dims=(hidden_size, 1), hidden_size=(2, 1))  # 2 = num_classes

    def forward(self, input_ids, attention_mask, labels=None):
        # Get hidden states from encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token output

        # Reshape to 3D (batch, features, 1) for NdLinear
        reshaped = cls_output.unsqueeze(-1)  # [batch_size, hidden_dim, 1]
        logits = self.ndlinear(reshaped).squeeze(-1)     # [batch_size, num_classes]

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

# Load accuracy metric
metric = evaluate.load("accuracy")

# Compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=15,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    logging_steps=10,
    do_eval=True,       # Enables evaluation
    logging_dir="./logs",  # Required for some older versions
)


model = NdLinearSentimentClassifier()

# Use small subset for fast testing
train_dataset = tokenized["train"].select(range(1000))
eval_dataset = tokenized["validation"].select(range(200))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
results = trainer.evaluate()
print(f"✅ Eval Accuracy: {results['eval_accuracy']:.4f}")

