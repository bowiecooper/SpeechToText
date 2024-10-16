import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_data, train_labels, epochs=100, lr=0.001):
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    for epoch in range(epochs):
        total_loss = 0
        for idx, (features, label) in enumerate(zip(train_data, train_labels)):
            features = torch.FloatTensor(features).unsqueeze(0)
            label = torch.LongTensor(label)
            
            optimizer.zero_grad()
            output = model(features)
            output_log_softmax = nn.functional.log_softmax(output, dim=-1)
            
            input_lengths = torch.full(size=(1,), fill_value=output.size(1), dtype=torch.long)
            target_lengths = torch.full(size=(1,), fill_value=len(label), dtype=torch.long)
            
            loss = criterion(output_log_softmax.transpose(0, 1), label.unsqueeze(0), input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            
            total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, File {idx+1}/{len(train_data)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_data)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")