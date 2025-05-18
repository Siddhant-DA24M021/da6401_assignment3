**Author**: Siddhant Baranwal (DA24M021)  
**GitHub**: [GITHUB LINK]()  
**Report**: [REPORT LINK]()



# DA6401-assignment-03-Siddhant-DA24M021
DA24M021 Siddhant Baranwal Assignment 3 of the course DA6401: Introduction to Deep Learning: Transliteration System with Seq2Seq

## Overview
This repository contains a sequence-to-sequence (Seq2Seq) model for transliterating Hindi words to Latin script. The implementation supports both vanilla Seq2Seq and attention-based architectures. The model is trained on the Dakshina dataset, focusing on Hindi transliteration.

## Dataset Setup
project_root/

├── dakshina_dataset_v1.0/ 

│   └── hi/

│       └── lexicons/

│           ├── hi.translit.sampled.train.tsv

│           ├── hi.translit.sampled.dev.tsv

│           └── hi.translit.sampled.test.tsv

## Dependencies
- Python 3.6+
- PyTorch 2.0+
- pandas
- tqdm
- matplotlib


# Install Dependencies

```bash
pip install torch pandas tqdm matplotlib
```


## Features
- **Model Variants**: Choose between `vanilla` Seq2Seq or `attention`-enhanced models.
- **Flexible Architecture**: Configure RNN cell type (LSTM, GRU, RNN), hidden layers, dropout, and more.
- **Training Metrics**: Track training/validation loss and accuracy during training.
- **Evaluation**: Calculate word-level accuracy on test data and transliterate custom inputs.
- **GPU Support**: Automatic CUDA detection for accelerated training.


# Using the model (Training and Evaluation)
python main.py \
  --model attention \          # Model type: vanilla/attention
  --batch_size 64 \            # Training batch size
  --learning_rate 0.001 \      # Learning rate
  --hidden_size 512 \          # RNN hidden size
  --num_layers 2 \             # Number of RNN layers
  --cell_type LSTM \           # RNN type: LSTM/GRU/RNN
  --dropout 0.2 \              # Dropout probability
  --num_epochs 20              # Training epochs


## Command Line Arguments

| Argument                 | Description                                   | Type    | Default | Choices/Options           |
|--------------------------|-----------------------------------------------|---------|---------|---------------------------|
| `-b`, `--batch_size`     | Batch size for training                       | int     | 32      | -                         |
| `-lr`, `--learning_rate` | Learning rate for optimizer                   | float   | 0.001   | -                         |
| `-es`, `--embedding_size`| Size of character embeddings                  | int     | 256     | -                         |
| `-hs`, `--hidden_size`   | Hidden dimension size for RNN layers          | int     | 512     | -                         |
| `-nl`, `--num_layers`    | Number of RNN layers                          | int     | 1       | -                         |
| `-ct`, `--cell_type`     | Type of RNN cell                              | str     | LSTM    | LSTM, GRU, RNN            |
| `-do`, `--dropout`       | Dropout probability                           | float   | 0.1     | -                         |
| `-m`, `--model`          | Model architecture type                       | str     | vanilla | vanilla, attention        |
| `-e`, `--num_epochs`     | Number of training epochs                     | int     | 10      | -                         |


# Self Declaration
I, Siddhant Baranwal DA24M021, swear on my honour that I have written the code and the report by myself and have not copied it from the internet or other students.