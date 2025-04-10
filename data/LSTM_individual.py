# 4. LSTM - 10개 종목별로 MSE, RMSE 값 분석

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Malgun Gothic', 'Segoe UI Emoji']
plt.rcParams['axes.unicode_minus'] = False

# 설정
TICKERS = ['HD', 'MCD', 'LOW', 'TJX', 'SBUX', 'NKE', 'MAR', 'CMG', 'ORLY', 'TGT']
SEQ_LEN = 20
BATCH_SIZE = 64
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 기술적 지표 10개 계산
def compute_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['ROC_10'] = df['Close'].pct_change(10) * 100
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['Stoch_%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stoch_%D'] = df['Stoch_%K'].rolling(3).mean()
    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_upper'] = ma20 + 2 * std20
    df['BB_lower'] = ma20 - 2 * std20
    tr = np.maximum(df['High'] - df['Low'],
                    np.maximum(abs(df['High'] - df['Close'].shift(1)),
                               abs(df['Low'] - df['Close'].shift(1))))
    df['ATR_14'] = tr.rolling(14).mean()
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    tp_ma = tp.rolling(14).mean()
    tp_std = tp.rolling(14).std()
    df['CCI_14'] = (tp - tp_ma) / (0.015 * tp_std)
    return df

# LSTM 모델
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Dataset 클래스
class StockDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X, self.y = [], []
        for i in range(len(X) - seq_len):
            self.X.append(X[i:i+seq_len])
            self.y.append(y[i+seq_len])
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# LSTM 성능 측정 (RMSE)
rmse_results = {}
FEATURES = ['SMA_20', 'EMA_20', 'RSI_14', 'ROC_10', 'Stoch_%K', 'Stoch_%D',
            'BB_upper', 'BB_lower', 'ATR_14', 'MACD', 'MACD_signal', 'CCI_14']

for ticker in TICKERS:
    print(f"\n📊 [종목: {ticker}] 시작")
    df = yf.download(ticker, period="2y", interval="1d", auto_adjust=False)
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
    df = compute_technical_indicators(df).dropna()

    X = df[FEATURES].values
    y = df['Adj Close'].values.reshape(-1, 1)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    dataset = StockDataset(X_scaled, y_scaled, SEQ_LEN)
    train_size = int(len(dataset) * 0.8)
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = StockLSTM(input_size=X.shape[1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

    # 평가
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb.to(DEVICE)).cpu().numpy()
            preds.append(pred)
            actuals.append(yb.numpy())
    preds = np.vstack(preds)
    actuals = np.vstack(actuals)
    preds_inv = scaler_y.inverse_transform(preds)
    actuals_inv = scaler_y.inverse_transform(actuals)
    rmse = np.sqrt(mean_squared_error(actuals_inv, preds_inv))
    rmse_results[ticker] = rmse
    print(f"✅ {ticker} RMSE: ${rmse:.2f}")

# 시각화
plt.figure(figsize=(10, 6))
plt.bar(rmse_results.keys(), rmse_results.values(), color='skyblue')
plt.title("📊 종목별 LSTM RMSE (기술 지표 10개)")
plt.ylabel("RMSE ($)")
plt.xlabel("Ticker")
plt.tight_layout()
plt.show()

### 추가 분석
### 10개 종목별 평균 주가 대비 RMSE 비율 측정 결과
import yfinance as yf

# 기존 RMSE 결과 수동 정의
rmse_results = {
    'MCD': 6.49, 'LOW': 6.73, 'TJX': 2.85, 'SBUX': 4.25,
    'NKE': 4.84, 'MAR': 8.65, 'CMG': 2.26, 'ORLY': 35.41, 'TGT': 6.07
}
TICKERS = list(rmse_results.keys())

rmse_percent_results = {}

for ticker in TICKERS:
    df = yf.download(ticker, period="2y", interval="1d", auto_adjust=False)
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
    mean_price = float(df['Adj Close'].mean())
    rmse = rmse_results[ticker]
    error_pct = (rmse / mean_price) * 100
    rmse_percent_results[ticker] = error_pct

print("\n📈 종목별 RMSE 및 평균 주가 대비 RMSE 비율:")
for ticker in TICKERS:
    print(f"{ticker}: RMSE = ${rmse_results[ticker]:.2f}, 평균가 대비 = {rmse_percent_results[ticker]:.2f}%")
