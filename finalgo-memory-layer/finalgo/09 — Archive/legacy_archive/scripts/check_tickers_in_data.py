import pandas as pd
import numpy as np

df = pd.read_csv('data/ranking_data_upstox.csv')

print('=== DayOfWeek Correlation Deep Dive ===')
print('DayOfWeek vs Next_Hour_Return mean by day:')
print(df.groupby('DayOfWeek')['Next_Hour_Return'].agg(['mean','count','std']))

print()
unique_qids = np.sort(df['Query_ID'].unique())
split_idx = int(len(unique_qids) * 0.8)
train_qids = unique_qids[:split_idx]
test_qids  = unique_qids[split_idx:]
train_df = df[df['Query_ID'].isin(train_qids)]
test_df  = df[df['Query_ID'].isin(test_qids)]

df['DateTime'] = pd.to_datetime(df['DateTime'])
train_df = df[df['Query_ID'].isin(train_qids)]
test_df  = df[df['Query_ID'].isin(test_qids)]

print('=== Train/Test Date Ranges ===')
print('Train period:', train_df['DateTime'].min(), 'to', train_df['DateTime'].max())
print('Test  period:', test_df['DateTime'].min(), 'to', test_df['DateTime'].max())

print()
print('=== DayOfWeek distribution: Train vs Test ===')
print('Train:', train_df['DayOfWeek'].value_counts().sort_index().to_dict())
print('Test :', test_df['DayOfWeek'].value_counts().sort_index().to_dict())

print()
print('=== Next_Hour_Return per DayOfWeek in Test only ===')
print(test_df.groupby('DayOfWeek')['Next_Hour_Return'].agg(['mean','count']))

print()
print('=== Checking for accidental Dist_52W_ train/test contamination ===')
print('Dist_52W_High stats - Train vs Test:')
print('  Train mean:', train_df['Dist_52W_High'].mean())
print('  Test  mean:', test_df['Dist_52W_High'].mean())
print('  Train std :', train_df['Dist_52W_High'].std())
print('  Test  std :', test_df['Dist_52W_High'].std())
