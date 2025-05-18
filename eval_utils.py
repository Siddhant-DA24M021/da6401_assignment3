import torch
from attention_seq2seq import Seq2SeqWithAttention

def evaluate(model, val_loader, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, tgt) in enumerate(val_loader):
            src = src.to(device)
            tgt = tgt.to(device)

            output = model.inference(src, tgt.shape[1])
            if isinstance(output, tuple): # For Attention Model
                output = output[0]

            # Reshape output and target for loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)

            # Calculate loss
            loss = criterion(output, tgt)
            epoch_loss += loss.item()

    return epoch_loss / len(val_loader)

def transliterate(model, src_text, src_vocab, tgt_vocab, device, max_length=100):
    model.eval()

    # Convert source text to tensor
    src_indices = src_vocab.encode(src_text)
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)

    # Get encoder outputs
    with torch.no_grad():
        if model.encoder.cell_type == 'LSTM':
            encoder_outputs, (hidden, cell) = model.encoder(src_tensor)
            decoder_hidden = (hidden, cell)
        else:
            encoder_outputs, hidden = model.encoder(src_tensor)
            decoder_hidden = hidden

    # Start with SOS token
    decoder_input = torch.tensor([tgt_vocab.char2idx[tgt_vocab.sos_token]], device=device)

    result_indices = [tgt_vocab.char2idx[tgt_vocab.sos_token]]

    for _ in range(max_length):
        with torch.no_grad():
            if isinstance(model, Seq2SeqWithAttention):
                decoder_output, decoder_hidden, attention_weights = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)

        # Get the most likely next character
        top_token = decoder_output.argmax(1).item()
        result_indices.append(top_token)

        # Stop if EOS token
        if top_token == tgt_vocab.char2idx[tgt_vocab.eos_token]:
            break

        # Use predicted token as next input
        decoder_input = torch.tensor([top_token], device=device)

    # Convert indices to text
    result_text = tgt_vocab.decode(result_indices, remove_special_tokens=True)

    return result_text

# Accuracy calculation function
def calculate_accuracy(model, data_loader, src_vocab, tgt_vocab, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            batch_size = src.shape[0]

            for i in range(batch_size):
                # Get source text and actual target text
                src_indices = src[i].tolist()
                src_text = src_vocab.decode(src_indices)
                actual_tgt_text = tgt_vocab.decode(tgt[i].tolist())

                # Get predicted transliteration
                predicted_tgt_text = transliterate(model, src_text, src_vocab, tgt_vocab, device)

                # Check if prediction matches
                if predicted_tgt_text == actual_tgt_text:
                    correct += 1
                total += 1

    return correct / total