import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader


print("Step 1: Data loading and preprocessing...")


class StockDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        df = df.dropna()
        self.stock_ids = df['stock_id'].unique()
        self.data = []
        self.labels = []
        for stock_id in self.stock_ids:
            stock_data = df[df['stock_id'] == stock_id].sort_values(by='time_id')
            features = stock_data.iloc[:, 2:-1].values
            label = stock_data['label'].values
            self.data.append(features)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)


print("Step 2: LSTM model definition...")


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)


def custom_loss(outputs, labels, alpha):
    mse_loss = nn.MSELoss()(outputs, labels)
    rank_loss = torch.sum(torch.max(torch.zeros_like(outputs) - (outputs - outputs.t()) * (labels - labels.t())))
    return mse_loss + alpha * rank_loss


print("Step 3: Training script...")


def train(model, dataloader, criterion, optimizer, alpha, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_features, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels, alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")
    torch.save(model.state_dict(), "/output/lstm_embedding_model.pth")


parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', type=int, default=8, help='Size of the embeddings')
parser.add_argument('--alpha', type=float, default=1.0, help='Weight for the ranking loss')
parser.add_argument('--epochs', type=int, default=10, help='Epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = StockDataset("/data/dyk2021/QuantContest/train_standard.csv")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    input_size = dataset[0][0].size(1)
    model = LSTMModel(input_size, args.embedding_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("Training the LSTM model...")
    train(model, dataloader, custom_loss, optimizer, args.alpha, num_epochs=args.epochs)
    print("Generating embeddings for each stock_id...")
    embeddings = {}
    model.eval()
    with torch.no_grad():
        for stock_id, (data, _) in zip(dataset.stock_ids, dataset):
            _, (hn, _) = model.lstm(data.unsqueeze(0))
            embeddings[stock_id] = hn[-1].squeeze().numpy()
    print("Saving embeddings dictionary...")
    np.save('/output/embeddings_dict.npy', embeddings)
