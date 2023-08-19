import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)


parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', type=int, default=8, help='Size of the embeddings')
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers')
parser.add_argument('--model_path', type=str, default='/output/lstm_embedding_model.pth', help='Path to load the trained model')
args = parser.parse_args()

print("Step 1: Prediction for test.csv and saving results to result.csv...")

def predict_and_save(model, test_file_path, model_path, output_file_path):
    # Loading trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Reading and preprocessing test data
    df_test = pd.read_csv(test_file_path)
    df_test = df_test.dropna()
    df_test = df_test.drop_duplicates(keep='first')

    # Preparing the data for model
    data_list = []
    for index, row in df_test.iterrows():
        features = torch.tensor(row[2:302].values, dtype=torch.float).unsqueeze(0)
        data_list.append(features)

    predictions = []

    # Making predictions for each row in test data
    with torch.no_grad():
        for data in data_list:
            output = model(data)
            predictions.append(output.item())

    # Adding the predictions to the test dataframe
    df_test['prediction'] = predictions

    # Saving required columns to result.csv
    result_df = df_test[['time_id', 'stock_id', 'prediction']]
    result_df.to_csv(output_file_path, index=False)

    print(f"Saved predictions to {output_file_path}")


if __name__ == '__main__':
    input_size = 300  # 300 features
    model = LSTMModel(input_size, args.embedding_size, num_layers=args.num_layers)
    # Predict and save the results
    predict_and_save(model, "/data/dyk2021/QuantContest/test.csv", args.model_path, '/output/result.csv')
