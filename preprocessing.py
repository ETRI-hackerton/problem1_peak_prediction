# preprocessing.py
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
# preprocessing.py

def preprocess_data(df):
    X = df[['tcp_pkt_count', 'udp_pkt_count', 'pkt_count', 'traffic_volume']]
    y = df['peak_1000ms'].dropna()  # NaN 제거

    X = X.loc[y.index]  # y가 NaN 아닌 것과만 정렬

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import TensorDataset
    import torch

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train.values).float())
    val_ds   = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val.values).float())
    test_ds  = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test.values).float())

    return train_ds, val_ds, test_ds, scaler
