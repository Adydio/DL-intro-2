import pandas as pd
import matplotlib.pyplot as plt
import json


df = pd.read_csv('/data/dyk2021/QuantContest/train.csv')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df_shifted = df.copy()
df_shifted['time_id'] = df_shifted['time_id'] + 2
merged_df = pd.merge(df[['time_id', 'stock_id', 'label']], df_shifted.drop('label', axis=1), on=['time_id', 'stock_id'], how='inner')
correlations = merged_df.iloc[:, 3:-1].corrwith(merged_df['label'])

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
