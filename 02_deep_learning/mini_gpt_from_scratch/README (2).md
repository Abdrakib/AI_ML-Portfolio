
# Mini GPT From Scratch (TinyStories)

## Overview
This project implements a GPT-style language model trained **from scratch** using the **TinyStories dataset**.
The goal of this project was to understand the full pipeline of training a transformer-based language model:
tokenizer creation, dataset preparation, model training, checkpointing, and text generation.

Unlike many projects that rely on pretrained models, this model was trained completely from scratch using PyTorch.

---

## Dataset
The model was trained using the TinyStories dataset.

Dataset source:
https://huggingface.co/datasets/eminorhan/tinystories

Characteristics:
- Short narrative stories
- Simple vocabulary
- Designed for small language model experiments

---

## Tokenizer
A tokenizer was trained and exported for the dataset.

Example tokenizer information:

- Vocabulary size: 16,000
- EOS token used for sequence termination
- Exported as `tokenizer.json`

Example loading code:

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
```

---

## Model Architecture
The model is a GPT-style transformer decoder implemented in PyTorch.

Main components:

- Token Embeddings
- Positional Embeddings
- Multi-Head Self-Attention
- Feedforward Layers
- Transformer Blocks
- Language Modeling Head

The model predicts the **next token in a sequence**.

---

## Training Setup
Training was performed using:

- PyTorch
- HuggingFace Datasets
- Google Colab GPU

Key features of the training pipeline:

- Checkpoint saving
- Validation loss monitoring
- Resume training capability
- Best checkpoint selection

Example training logs:

```
step 76000 | train_loss ...
step 78000 | val_loss ...
Saved LAST.pt
```

---

## Results
The model successfully learned:

- Story structure
- Narrative patterns
- Basic sentence continuation

Example generated output:

PROMPT: Once upon a time

OUTPUT:
Once upon a time there was a boy who liked to play near the beach...

---

## Limitations
Training a language model from scratch with limited compute resources introduces some limitations:

- Imperfect grammar
- Occasionally fragmented tokens
- Limited fluency compared to large pretrained models

These results highlight the challenges of training language models from scratch.

---

## What I Learned
Through this project I learned:

- How GPT-style transformers work
- Tokenization and vocabulary building
- Dataset preparation for language models
- Training loops and checkpointing
- Text generation and sampling methods

---

## Future Improvements
Possible improvements include:

- Larger model architecture
- Longer training
- Improved tokenizer
- Fine-tuning on specialized datasets

---

## Project Structure

mini-gpt-from-scratch/

├── Mini_LLM_TRAINING.ipynb  
├── README.md  
├── requirements.txt  

└── screenshots/  
    ├── tokenizer.png  
    ├── dataset.png  
    ├── training_logs.png  
    ├── best_model.png  
    └── generation_output.png  

---

## Technologies Used

- Python
- PyTorch
- HuggingFace Datasets
- HuggingFace Tokenizers
- Google Colab
