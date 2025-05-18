import torch
import argparse
from data_utils import get_vocab, get_dataloader
from train_utils import train
from eval_utils import evaluate, transliterate, calculate_accuracy
from tqdm import tqdm


# Data File Paths
TRAIN_FilePath = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv'
DEV_FilePath = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv'
TEST_FilePath = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv'

# Accelerator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--batch_size", type = int, default = 32, help = "Batch size used to train neural network.")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 0.001, help = "Learning rate used to optimize model parameters")
    parser.add_argument("-es", "--embedding_size", type = int, default = 256, help = "Size of the embedding layer.")
    parser.add_argument("-hs", "--hidden_size", type = int, default = 512, help = "Size of the hidden layer.")
    parser.add_argument("-nl", "--num_layers", type = int, default = 1, help = "Number of layers in the RNN.")
    parser.add_argument("-ct", "--cell_type", type = str, default = "LSTM", choices=["LSTM", "GRU", "RNN"], help = "Type of RNN cell to use.")
    parser.add_argument("-do", "--dropout", type = float, default = 0.1, help = "Dropout rate for the model.")
    parser.add_argument("-m", "--model", type = str, default = "vanilla", choices=["vanilla", "attention"], help = "Type of model to use.")
    parser.add_argument("-e", "--num_epochs", type = int, default = 10, help = "Number of epochs to train the model.")


    return parser.parse_args()

def main():
    args = parse_arguments()

    src_vocab, tgt_vocab = get_vocab(TRAIN_FilePath)

    # Get Dataloaders
    train_loader = get_dataloader(TRAIN_FilePath, src_vocab=src_vocab, tgt_vocab=tgt_vocab, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(DEV_FilePath, src_vocab=src_vocab, tgt_vocab=tgt_vocab, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(TEST_FilePath, src_vocab=src_vocab, tgt_vocab=tgt_vocab, batch_size=args.batch_size, shuffle=False)

    INPUT_SIZE = src_vocab.vocab_size
    OUTPUT_SIZE = tgt_vocab.vocab_size
    EMBEDDING_SIZE = args.embedding_size
    HIDDEN_SIZE = args.hidden_size
    NUM_LAYERS = args.num_layers
    CELL_TYPE = args.cell_type
    DROPOUT = args.dropout
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.num_epochs

    # Initialize Model
    if args.model == "vanilla":
        from vanilla_seq2seq import Encoder, Decoder, Seq2Seq

        encoder = Encoder(
            input_size=INPUT_SIZE,
            embedding_size=EMBEDDING_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            cell_type=CELL_TYPE,
            dropout=DROPOUT
        )

        decoder = Decoder(
            output_size=OUTPUT_SIZE,
            embedding_size=EMBEDDING_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            cell_type=CELL_TYPE,
            dropout=DROPOUT
        )

        model = Seq2Seq(encoder, decoder, device).to(device)

    elif args.model == "attention":
        from attention_seq2seq import Encoder, Attention, AttentionDecoder, Seq2SeqWithAttention

        encoder = Encoder(
            input_size=INPUT_SIZE,
            embedding_size=EMBEDDING_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            cell_type=CELL_TYPE,
            dropout=DROPOUT
        )

        attention = Attention(HIDDEN_SIZE)

        decoder = AttentionDecoder(
            output_size=OUTPUT_SIZE,
            embedding_size=EMBEDDING_SIZE,
            hidden_size=HIDDEN_SIZE,
            attention=attention,
            num_layers=NUM_LAYERS,
            cell_type=CELL_TYPE,
            dropout=DROPOUT
        )

        model = Seq2SeqWithAttention(encoder, decoder, attention, device).to(device)

    # Initialize Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_vocab.char2idx[tgt_vocab.pad_token])

    # Training Loop
    for epoch in tqdm(range(NUM_EPOCHS)):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    train_accuracy = calculate_accuracy(model, train_loader, src_vocab, tgt_vocab, device)
    val_accuracy = calculate_accuracy(model, val_loader, src_vocab, tgt_vocab, device)
    test_accuracy = calculate_accuracy(model, test_loader, src_vocab, tgt_vocab, device)

    print(f"Train Accuracy : {train_accuracy*100:6.2f}%")
    print(f"Val Accuracy   : {val_accuracy*100:6.2f}%")
    print(f"Test Accuracy  : {test_accuracy*100:6.2f}%")

if __name__ == "__main__":
    main()