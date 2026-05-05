import pandas as pd

DATA_PATH = 'data/kairognos_pre_breakout_under5k_vol500k.parquet'

df = pd.read_parquet(DATA_PATH)
print(f'Shape: {df.shape}')
print(df.head())
