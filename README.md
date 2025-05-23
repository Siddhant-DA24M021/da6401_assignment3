**Author**: Siddhant Baranwal (DA24M021)  
**GitHub**: [GITHUB LINK](https://github.com/Siddhant-DA24M021/da6401_assignment3.git)  
**Report**: [REPORT LINK](https://wandb.ai/da24m021-indian-institute-of-technology-madras/da24m021_da6401_assignment3/reports/DA6401-Assignment-3-DA24M021---VmlldzoxMjc5MDEyNg?accessToken=4mbvpuvktagn9rcbvixus7wwbwrpazy9fk712u370f1n6yfc639jffnkt0obu1or)
**Alternate Report Link**: [ALTERNATE REPORT LINK](https://api.wandb.ai/links/da24m021-indian-institute-of-technology-madras/uv4jucjo)



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

## Files and Folder

- da6401_assignment3.ipynb : Original .ipynb file

- main.py : Training and Evaluating file

- vanilla_seq2seq.py : Vanilla seq2seq model file
- attention_seq2seq.py : Attention seq2seq model file

- data_utils.py : File contains data loading and preprocessing codes
- train_utils.py : Training code for the models
- eval_utils.py : Evaluating and transliteration codes

- predictions_vanilla : Contains predictions on test dataset using vanilla seq2seq model 
- predictions_attention : Contains predictions on test dataset using attention seq2seq model


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

**IMPORTANT:- Add dataset and update dataset paths in the main.py.**
python main.py \
  --model attention \          # Model type: vanilla/attention
  --batch_size 64 \            # Training batch size
  --learning_rate 0.001 \      # Learning rate
  --hidden_size 512 \          # RNN hidden size
  --num_layers 2 \             # Number of RNN layers
  --cell_type LSTM \           # RNN type: LSTM/GRU/RNN
  --dropout 0.2 \              # Dropout probability
  --num_epochs 20              # Training epochs

``` bash
  python main.py --model attention --batch_size 64 --learning_rate 0.001 --hidden_size 512 --num_layers 2 --cell_type LSTM --dropout 0.2  --num_epochs 2
```


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