import torch

def train(model, train_loader, optimizer, criterion, device, clip=1.0, teacher_forcing_ratio=0.5):
    model.train()
    epoch_loss = 0

    for i, (src, tgt) in enumerate(train_loader):
        src = src.to(device)
        tgt = tgt.to(device)
        #print("tgt", tgt)
        optimizer.zero_grad()

        output = model(src, tgt, teacher_forcing_ratio)
        #print("output", output)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)
        #print("tgt2", tgt)
        #print("output2", output)
        #break

        # Calculate loss
        loss = criterion(output, tgt)

        # Backpropagation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Update parameters
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)

