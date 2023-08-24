import pandas as pd
import matplotlib.pyplot as plt
import json

df = pd.read_csv('/data/dyk2021/QuantContest/train.csv')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)


def replace_features(row):
    target_time = row['time_id'] + 2
    target_stock = row['stock_id']
    target_row = df[(df['time_id'] == target_time) & (df['stock_id'] == target_stock)]
    if target_row.empty:
        return None
    else:
        for col in range(300):
            row[str(col)] = target_row[str(col)].values[0]
        return row


df1 = df.apply(replace_features, axis=1)
df1.dropna(inplace=True)

correlations = df1.iloc[:, 2:-1].corrwith(df1['label'])

plt.figure(figsize=(10, 5))
correlations.plot(kind='bar')
plt.title('Feature Correlations with Label')
plt.xlabel('Feature Number')
plt.ylabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('/output/correlation_plot.png')
plt.show()

with open('/output/correlations.json', 'w') as json_file:
    json.dump(correlations.to_dict(), json_file)
