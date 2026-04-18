import pandas as pd

DATA_PATH = 'data/kairognos_ml_dataset_with_mktcap.parquet'

df = pd.read_parquet(DATA_PATH)
print(f'Shape: {df.shape}')
print(df.head())
