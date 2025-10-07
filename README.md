# Turkish NER Fine-Tuned Model

## Description
This repository contains a **BERT-based Turkish Named Entity Recognition (NER) model** fine-tuned on the [erayyildiz/turkish_ner](https://huggingface.co/datasets/erayyildiz/turkish_ner) dataset.  
The base model used for fine-tuning is [`dbmdz/bert-base-turkish-w`](https://huggingface.co/dbmdz/bert-base-turkish-w). 

## Use my fine-tuned Model: [https://huggingface.co/gokhanErgul/turkish-ner-outpu](https://huggingface.co/gokhanErgul/turkish-ner-output)

After fine-tuning for 1 epoch, the model achieved the following metrics on the validation set:

- **Training Loss:** 0.2969  
- **Validation Loss:** 0.2948  
- **Precision:** 0.6473  
- **Recall:** 0.6699  
- **F1-score:** 0.6584  
- **Accuracy:** 0.8835  


---

## 🧰 Repository Structure
- `Explanation.ipynb` – Detailed explanations of data preprocessing & label alignment  
- `Token Classification.ipynb` – Notebook used for training (example code & outputs)  
- `README.md`  
- `requirements.txt`

---

## 📓 Explanation.ipynb

This notebook contains detailed explanations of complex preprocessing steps, especially token-label alignment and dataset mapping.  

Since the dataset is already word-tokenized, BERT tokenizer outputs are aligned with `word_ids()`.  
Details include updating "B-XXX" labels to "I-XXX" for subtokens, among other alignment rules.

---

## 🚀 Usage

You can quickly use the model with Hugging Face’s pipeline:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("output_ner")
model = AutoModelForTokenClassification.from_pretrained("output_ner")

# Create NER pipeline
ner_pipe = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=0  # Uses GPU if available
)

# Example sentence
sentence = "Hello, I am Gökhan and I live in Istanbul."
results = ner_pipe(sentence)

for r in results:
    print(r)

