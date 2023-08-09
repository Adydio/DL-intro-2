import json
import argparse


parser = argparse.ArgumentParser(description='RNN for Stock Feature Extraction')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
parser.add_argument('--epochs', type=int, default=10, help='Num of epochs')
parser.add_argument('--hidden', type=int, default=128, help='Hidden size')
args = parser.parse_args()


import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os


print("Starting data preprocessing...")


data_path = "/data/dyk2021/QuantContest/train.csv"
data = pd.read_csv(data_path)
data.dropna(inplace=True)


class StockDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data['time_id'].unique())

    def __getitem__(self, idx):
        time_id = self.data['time_id'].unique()[idx]
        time_data = self.data[self.data['time_id'] == time_id]
        time_data_sorted = time_data.sort_values(by='stock_id')
        features = time_data_sorted.iloc[:, 2:-1].values
        labels = time_data_sorted['label'].values
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

dataset = StockDataset(data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


print("Building RNN model...")

class StockRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(StockRNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        out, hidden = self.rnn(x)
        fc = nn.Linear(self.rnn.hidden_size, x.size(1)).to(x.device)
        out = fc(out[:, -1, :])
        return out, hidden


input_size = 300
hidden_size = args.hidden
num_layers = 2
model = StockRNN(input_size, args.hidden, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


print("Starting training...")

for epoch in range(args.epochs):
    total_loss = 0.0
    num_batches = 0

    for i, (features, labels) in enumerate(dataloader):
        outputs, _ = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Epoch [{epoch + 1}/{args.epochs}], Average Loss: {avg_loss:.10f}")


    if (epoch + 1) % 10 == 0:
        model_save_path = f"/output/lr_{args.lr}_total_{args.epochs}_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at epoch {epoch + 1} to {model_save_path}")



print("Saving model...")
model_path = "/output/stock_rnn_model.pth"
torch.save(model.state_dict(), model_path)


print("Extracting feature vectors...")
model.eval()
feature_dict = {}
with torch.no_grad():
    for i, (features, _) in enumerate(dataloader):
        _, hidden = model(features)
        time_id = int(data['time_id'].unique()[i])
        feature_vector = hidden[-1].squeeze().numpy()
        feature_dict[time_id] = feature_vector.tolist()


print("Saving feature vectors mapping...")
mapping_path = "/output/rnn.json"
with open(mapping_path, 'w') as f:
    json.dump(feature_dict, f)

print("Process completed!")
