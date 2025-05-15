import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=1, cell_type="RNN", dropout=0.0):
        super().__init__()

        if num_layers == 1:
            dropout = 0

        self.cell_type = cell_type

        # Embedding Layer
        self.embedding = nn.Embedding(input_size, embedding_size)

        # Recurrent Layer
        if cell_type == "LSTM":
            self.recurrent_layer = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        elif cell_type == "GRU":
            self.recurrent_layer = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        else: # Default (RNN)
            self.recurrent_layer = nn.RNN(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input: batch_size x seq_len

        embeddings = self.embedding(input)
        embeddings = self.dropout(embeddings)

        if self.cell_type == 'LSTM':
            outputs, (hidden, cell) = self.recurrent_layer(embeddings)
            return outputs, (hidden, cell)
        else:
            outputs, hidden = self.recurrent_layer(embeddings)
            return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers=1, cell_type="RNN", dropout=0.0):
        super().__init__()

        if num_layers == 1:
            dropout = 0

        self.output_size = output_size
        self.cell_type = cell_type

        # Embedding Layer
        self.embedding = nn.Embedding(output_size, embedding_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Recurrent Layer
        if cell_type == "LSTM":
            self.recurrent_layer = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        elif cell_type == "GRU":
            self.recurrent_layer = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        else: # Default (RNN)
            self.recurrent_layer = nn.RNN(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):

        input = input.unsqueeze(1)
        embeddings = self.embedding(input)
        embeddings = self.dropout(embeddings)

        if self.cell_type == "LSTM":
            hidden, cell = hidden
            outputs, (hidden, cell) = self.recurrent_layer(embeddings, (hidden, cell))
            hidden = (hidden, cell)
        else:
            outputs, hidden = self.recurrent_layer(embeddings, hidden)

        outputs = outputs.squeeze(1)
        prediction = self.fc_out(outputs)
        return prediction, hidden
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):

        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_size

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        if self.encoder.cell_type == 'LSTM':
            encoder_outputs, (hidden, cell) = self.encoder(src)
            decoder_hidden = (hidden, cell)
        else:
            encoder_outputs, hidden = self.encoder(src)
            decoder_hidden = hidden

        decoder_input = tgt[:, 0]
        for t in range(1, tgt_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            top = decoder_output.argmax(1)
            decoder_input = tgt[:, t] if teacher_force else top
        return outputs


    def inference(self, src, max_len, sos_idx=1, eos_idx=2):

        batch_size = src.shape[0]
        tgt_len = max_len
        tgt_vocab_size = self.decoder.output_size
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        if self.encoder.cell_type == 'LSTM':
            encoder_outputs, (hidden, cell) = self.encoder(src)
            decoder_hidden = (hidden, cell)
        else:
            encoder_outputs, hidden = self.encoder(src)
            decoder_hidden = hidden

        decoder_input = torch.tensor([sos_idx] * batch_size, device=self.device)
        for t in range(1, max_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t] = decoder_output
            top = decoder_output.argmax(1)
            decoder_input = top

            # Check if all sequences have reached <eos>
            if (outputs == eos_idx).any(dim=1).all():
                break
        return outputs