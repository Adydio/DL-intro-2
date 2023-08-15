import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import argparse
from torch.utils.data import Dataset, DataLoader


print("Loading mean and variance...")


with open("./output/mean_variance_dict.json", "r") as file:
    mv_dict = json.load(file)
means = mv_dict['mean']
vars = mv_dict['var']


class TestStockDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        df = df.dropna()
        for i in range(300):
            df[str(i)] = (df[str(i)] - means[i]) / np.sqrt(vars[i])
        self.stock_ids = df['stock_id'].unique()
        self.data = []
        self.time_ids = []
        for stock_id in self.stock_ids:
            stock_data = df[df['stock_id'] == stock_id].sort_values(by='time_id')
            features = stock_data.iloc[:, 2:].values  # Exclude time_id and stock_id columns
            self.data.append(features)
            self.time_ids.append(stock_data['time_id'].values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float), self.time_ids[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)


parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', type=int, default=8, help='Size of the embeddings')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = TestStockDataset("/data/dyk2021/QuantContest/test.csv")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    input_size = dataset[0][0].size(1)
    model = LSTMModel(input_size, args.embedding_size)
    model.load_state_dict(torch.load("/output/lstm_embedding_model.pth"))
    model.eval()
    results = []
    with torch.no_grad():
        for (data, time_ids), stock_id in zip(dataloader, dataset.stock_ids):
            predictions = model(data).squeeze().numpy()
            if np.ndim(predictions) == 0:
                predictions = np.array([predictions])
            for time_id, prediction in zip(time_ids[0], predictions):
                results.append((time_id, stock_id, prediction))
    sorted_results = sorted(results, key=lambda x: (x[0], x[1]))
    result_df = pd.DataFrame(sorted_results, columns=['time_id', 'stock_id', 'prediction'])
    label_df = pd.read_csv('/data/dyk2021/QuantContest/test_label_not_null.csv')
    result_df['label'] = label_df['label']
    result_df.to_csv('result.csv', index=False)

print("Predictions saved to result.csv.")
