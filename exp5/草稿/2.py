import pandas as pd
import matplotlib.pyplot as plt
import json


df = pd.read_csv('/data/dyk2021/QuantContest/train.csv')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

rows_list = []
feature_cols = [str(i) for i in range(300)]

for index, row in df.iterrows():
    time_id_future = row['time_id'] + 2
    stock_id = row['stock_id']

    future_data = df[(df['time_id'] == time_id_future) & (df['stock_id'] == stock_id)]

    if not future_data.empty:
        diff_features = future_data[feature_cols].values - row[feature_cols].values
        new_row = {'time_id': row['time_id'], 'stock_id': stock_id, 'label': row['label']}
        for i, feat in enumerate(feature_cols):
            new_row[feat] = diff_features[0][i]
        rows_list.append(new_row)


df1 = pd.DataFrame(rows_list)

correlation = df1[feature_cols + ['label']].corr()['label'].drop('label')

correlation.to_json("/output/correlation.json")


plt.figure(figsize=(14, 7))
correlation.plot(kind='bar')
plt.title("Correlation between features and label")
plt.ylabel("Correlation coefficient")
plt.xlabel("Features")
plt.tight_layout()
plt.savefig("/output/correlation_plot.png")
