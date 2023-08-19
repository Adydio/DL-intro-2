import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import json
from torch.utils.data import Dataset, DataLoader

# 为简单起见，定义一个 UNK_TOKEN
UNK_TOKEN = [0.0] * 300

print("Step 1: Data loading and preprocessing...")


class StockDataset(Dataset):
    def __init__(self, data_path, mask_prob):
        df = pd.read_csv(data_path)
        df = df.dropna()
        df = df.drop_duplicates(keep='first')
        self.stock_ids = df['stock_id'].unique().tolist()
        self.stock_ids.append('<unk>')  # 添加一个特殊的 stock_id
        self.data = []
        self.labels = []
        for stock_id in self.stock_ids:
            if stock_id == '<unk>':
                features = [UNK_TOKEN]
                label = [0.0]
            else:
                stock_data = df[df['stock_id'] == stock_id].sort_values(by='time_id')
                features = stock_data.iloc[:, 2:-1].values
                label = stock_data['label'].values

                # Masking with a certain probability
                if np.random.rand() < mask_prob:
                    features = [UNK_TOKEN for _ in range(features.shape[0])]
            self.data.append(features)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)


print("Step 2: LSTM model definition...")


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
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


parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', type=int, default=8, help='Size of the embeddings')
parser.add_argument('--alpha', type=float, default=0, help='alpha')
parser.add_argument('--mask_prob', type=float, default=0.1, help='Probability of masking a stock_id to <unk>')
parser.add_argument('--epochs', type=int, default=10, help='Epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers')
parser.add_argument('--standard', type=int, default=0, help='1 or 0')
args = parser.parse_args()

if __name__ == '__main__':
    if args.standard == 1:
        dataset = StockDataset("/data/dyk2021/QuantContest/train_standard.csv", args.standard)
    else:
        dataset = StockDataset("/data/dyk2021/QuantContest/train.csv", args.standard)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    input_size = dataset[0][0].size(1)
    print(f"input size:{input_size}")
    model = LSTMModel(input_size, args.embedding_size, num_layers=args.num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("Training the LSTM model...")
    train(model, dataloader, custom_loss, optimizer, args.alpha, num_epochs=args.epochs)
    print("Generating embeddings for each stock_id...")
    embeddings = {}
    model.eval()
    with torch.no_grad():
        for stock_id, (data, _) in zip(dataset.stock_ids, dataset):
            _, (hn, _) = model.lstm(data.unsqueeze(0))
            embeddings[stock_id] = hn[-1].squeeze().numpy().tolist()
    print("Saving embeddings dictionary...")
    np.save('/output/embeddings_dict.npy', embeddings)
    with open('/output/embeddings_dict.json', 'w') as json_file:
        json.dump(embeddings, json_file)
    torch.save(embeddings, '/output/embeddings_dict.pt')
