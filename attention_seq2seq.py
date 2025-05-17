import torch
import torch.nn as nn
import torch.nn.functional as F
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
        

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        # Attention parameters for: e_jt = Vatt^T tanh(Uatt * s_t-1 + Watt * c_j)
        self.Uatt = nn.Linear(hidden_size, hidden_size)
        self.Watt = nn.Linear(hidden_size, hidden_size)
        self.Vatt = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs):

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        if isinstance(hidden, tuple):  # LSTM
            hidden_state = hidden[0]
            hidden_state = hidden_state[-1]
        else:
            hidden_state = hidden[-1]

        hidden_expanded = hidden_state.unsqueeze(1)

        # Uatt * s_t-1
        uatt_term = self.Uatt(hidden_expanded)

        # Watt * c_j
        watt_term = self.Watt(encoder_outputs)


        combined = torch.tanh(uatt_term + watt_term)

        # Apply Vatt^T
        energy = self.Vatt(combined)
        energy = energy.squeeze(2)

        attention_weights = F.softmax(energy, dim=1)

        # Create context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)

        return context, attention_weights
    
class AttentionDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, attention, num_layers=1, cell_type="RNN", dropout=0.0):
        super().__init__()

        if num_layers == 1:
            dropout = 0

        self.output_size = output_size
        self.cell_type = cell_type
        self.attention = attention
        self.hidden_size = hidden_size

        # Embedding Layer
        self.embedding = nn.Embedding(output_size, embedding_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        if cell_type == "LSTM":
            self.recurrent_layer = nn.LSTM(embedding_size + hidden_size, hidden_size, num_layers,
                                         dropout=dropout, batch_first=True)
        elif cell_type == "GRU":
            self.recurrent_layer = nn.GRU(embedding_size + hidden_size, hidden_size, num_layers,
                                        dropout=dropout, batch_first=True)
        else: # Default (RNN)
            self.recurrent_layer = nn.RNN(embedding_size + hidden_size, hidden_size, num_layers,
                                        dropout=dropout, batch_first=True)

        # Output layer
        self.fc_out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, hidden, encoder_outputs):

        input = input.unsqueeze(1)
        embeddings = self.embedding(input)
        embeddings = self.dropout(embeddings)

        # Calculate attention context vector
        context, attention_weights = self.attention(hidden, encoder_outputs)

        # Concatenate embeddings and context vector
        context = context.unsqueeze(1)
        rnn_input = torch.cat((embeddings, context), dim=2)

        if self.cell_type == "LSTM":
            outputs, (hidden, cell) = self.recurrent_layer(rnn_input, hidden)
            hidden_state = hidden
            hidden_tuple = (hidden, cell)
        else:
            outputs, hidden = self.recurrent_layer(rnn_input, hidden)
            hidden_state = hidden
            hidden_tuple = hidden

        if isinstance(hidden_state, tuple):  # LSTM
            last_hidden = hidden_state[0][-1]
        else:
            last_hidden = hidden_state[-1]

        last_hidden = last_hidden.squeeze(0) if last_hidden.dim() > 2 else last_hidden

        outputs = outputs.squeeze(1)
        context = context.squeeze(1)

        output_vector = torch.cat((outputs, context), dim=1)
        prediction = self.fc_out(output_vector)

        return prediction, hidden_tuple, attention_weights
    
class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, attention, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]

        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_size

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # Store attention weights for visualization
        attentions = torch.zeros(batch_size, tgt_len, src.shape[1]).to(self.device)

        # Encode the source sequence
        if self.encoder.cell_type == 'LSTM':
            encoder_outputs, (hidden, cell) = self.encoder(src)
            decoder_hidden = (hidden, cell)
        else:
            encoder_outputs, hidden = self.encoder(src)
            decoder_hidden = hidden

        # First input to the decoder is the <sos> token
        decoder_input = tgt[:, 0]

        # Start decoding
        for t in range(1, tgt_len):
            # Decode
            decoder_output, decoder_hidden, attention_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            # Store decoder output and attention
            outputs[:, t] = decoder_output
            attentions[:, t] = attention_weights

            # Teacher forcing: use ground truth or predicted token as next input
            teacher_force = random.random() < teacher_forcing_ratio
            top = decoder_output.argmax(1)
            decoder_input = tgt[:, t] if teacher_force else top

        return outputs, attentions

    def inference(self, src, max_len, sos_idx=1, eos_idx=2):
        # src: [batch_size, src_len]
        batch_size = src.shape[0]

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, max_len, self.decoder.output_size).to(self.device)

        # Store attention weights for visualization
        attentions = torch.zeros(batch_size, max_len, src.shape[1]).to(self.device)

        # Encode the source sequence
        if self.encoder.cell_type == 'LSTM':
            encoder_outputs, (hidden, cell) = self.encoder(src)
            decoder_hidden = (hidden, cell)
        else:
            encoder_outputs, hidden = self.encoder(src)
            decoder_hidden = hidden

        # First input to the decoder is the <sos> token
        decoder_input = torch.tensor([sos_idx] * batch_size, device=self.device)

        # Start decoding
        for t in range(1, max_len):
            # Decode
            decoder_output, decoder_hidden, attention_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            # Store decoder output and attention
            outputs[:, t] = decoder_output
            attentions[:, t] = attention_weights

            # Use predicted token as next input
            top = decoder_output.argmax(1)
            decoder_input = top

            # Check if all sequences have reached <eos>
            if (top == eos_idx).all():
                break

        return outputs, attentions
