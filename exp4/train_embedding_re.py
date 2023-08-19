import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader


class StockDataset(Dataset):
    def __init__(self, data_path, n):
        df = pd.read_csv(data_path)
        df = df.dropna()
        df = df.drop_duplicates(keep='first')
        self.stock_ids = df['stock_id'].unique()
        self.data = []
        self.labels = []
        self.max_length = df.groupby('stock_id').size().max()  # Max length for padding

        for stock_id in self.stock_ids:
            stock_data = df[df['stock_id'] == stock_id].sort_values(by='time_id')
            if n == 1:
                features = stock_data.iloc[:, 3:-1].values
            else:
                features = stock_data.iloc[:, 2:-1].values
            label = stock_data['label'].values

            # Padding
            padding_length = self.max_length - len(features)
            features = np.pad(features, [(0, padding_length), (0, 0)], mode='constant')
            label = np.pad(label, [(0, padding_length)], mode='constant')

            self.data.append(features)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)


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
    if alpha > 0:
        rank_loss = torch.sum(torch.max(torch.zeros_like(outputs) - (outputs - outputs.t()) * (labels - labels.t())))
    else:
        rank_loss = 0
    return mse_loss + alpha * rank_loss


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
parser.add_argument('--alpha', type=float, default=0, help='Weight for the ranking loss')
parser.add_argument('--epochs', type=int, default=10, help='Epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size for training')
parser.add_argument('--standard', type=int, default=1, help='1 or 0')
parser.add_argument('--base_dir', type=str, default="/data/dyk2021/QuantContest")
parser.add_argument('--load_path', type=str, default='',
                    help='Path to the pre-trained model. If provided, training is skipped.')

args = parser.parse_args()

if __name__ == '__main__':
    if args.standard == 1:
        train_file_standard = args.base_dir + "/train_standard.csv"
        dataset = StockDataset(train_file_standard, args.standard)
    else:
        train_file = args.base_dir + "/train.csv"
        dataset = StockDataset(train_file, args.standard)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            drop_last=True)  # drop_last to avoid dimension issues
    input_size = dataset[0][0].size(1)
    model = LSTMModel(input_size, args.embedding_size, num_layers=args.num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.load_path == '':
        print("Training the LSTM model...")
        train(model, dataloader, custom_loss, optimizer, args.alpha, num_epochs=args.epochs)
    else:
        print(f"Loading model from {args.load_path}...")
        model.load_state_dict(torch.load(args.load_path))

    print("Generating embeddings for each stock_id...")
    embeddings = {}
    model.eval()
    with torch.no_grad():
        for stock_id, (data, _) in zip(dataset.stock_ids, dataset):
            _, (hn, _) = model.lstm(data.unsqueeze(0))
            embeddings[stock_id] = hn[-1].squeeze().numpy()

    print("Saving embeddings dictionary...")
    np.save('/output/embeddings_dict_.npy', embeddings)
