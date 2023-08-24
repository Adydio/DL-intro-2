import pandas as pd
import matplotlib.pyplot as plt
import json


df = pd.read_csv('/data/dyk2021/QuantContest/train.csv')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['time_id_future'] = df['time_id'] + 2
merged_df = pd.merge(df, df, left_on=['stock_id', 'time_id_future'], right_on=['stock_id', 'time_id'], suffixes=('', '_future'))
feature_cols = [str(i) for i in range(300)]
for col in feature_cols:
    merged_df[col] = merged_df[col + '_future'] - merged_df[col]
df1 = merged_df[['time_id', 'stock_id'] + feature_cols + ['label']]
correlation = df1[feature_cols + ['label']].corr()['label'].drop('label')
correlation.to_json("/output/correlation_2.json")

plt.figure(figsize=(14, 7))
correlation.plot(kind='bar')
plt.title("Correlation between features and label")
plt.ylabel("Correlation coefficient")
plt.xlabel("Features")
plt.tight_layout()
plt.savefig("/output/correlation_plot_2.png")
