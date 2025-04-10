# 5. LSTM으로 통합 데이터셋에서 기술적 지표 조합별 성능 비교
## 기술적 지표 6개 정도로 추려짐.

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Malgun Gothic', 'Segoe UI Emoji']
plt.rcParams['axes.unicode_minus'] = False

# ========== 설정 ==========
TICKERS = ['HD', 'MCD', 'LOW', 'TJX', 'SBUX', 'NKE', 'MAR', 'CMG', 'ORLY', 'TGT']
PERIOD = "2y"
INTERVAL = "1d"
SEQ_LEN = 20
BATCH_SIZE = 64
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TECHNICAL_FEATURES = ['SMA_20', 'EMA_20', 'RSI_14', 'ROC_10', 'Stoch_%K', 'Stoch_%D',
#                       'BB_upper', 'BB_lower', 'ATR_14', 'MACD', 'MACD_signal', 'CCI_14', 'OBV']
TECHNICAL_FEATURES = ['SMA_20', 'EMA_20','BB_upper', 'BB_lower', 'OBV', 'Stoch_%K', 'MACD', 'MACD_signal']

# ========== 기술적 지표 계산 함수 ==========
def compute_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    ma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_upper'] = ma20 + 2 * std20
    df['BB_lower'] = ma20 - 2 * std20
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['OBV'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0))
    df['OBV'] = df['OBV'].cumsum()
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df
# # ========== 기술적 지표 계산 함수 ==========
# def compute_technical_indicators(df):
#     df['SMA_20'] = df['Close'].rolling(window=20).mean()
#     df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
#     delta = df['Close'].diff()
#     gain = delta.clip(lower=0)
#     loss = -delta.clip(upper=0)
#     avg_gain = gain.rolling(window=14).mean()
#     avg_loss = loss.rolling(window=14).mean()
#     rs = avg_gain / avg_loss
#     df['RSI_14'] = 100 - (100 / (1 + rs))
#     df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
#     low_14 = df['Low'].rolling(window=14).min()
#     high_14 = df['High'].rolling(window=14).max()
#     df['Stoch_%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
#     df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()
#     ma20 = df['Close'].rolling(window=20).mean()
#     std20 = df['Close'].rolling(window=20).std()
#     df['BB_upper'] = ma20 + 2 * std20
#     df['BB_lower'] = ma20 - 2 * std20
#     tr = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
#     df['ATR_14'] = tr.rolling(window=14).mean()
#     ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
#     ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
#     df['MACD'] = ema_12 - ema_26
#     df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
#     tp = (df['High'] + df['Low'] + df['Close']) / 3
#     tp_ma = tp.rolling(window=14).mean()
#     tp_std = tp.rolling(window=14).std()
#     df['CCI_14'] = (tp - tp_ma) / (0.015 * tp_std)
#     df['OBV'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0))
#     df['OBV'] = df['OBV'].cumsum()
#     return df

# ========== LSTM 모델 정의 ==========
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ========== PyTorch Dataset ==========
class StockDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X, self.y = [], []
        for i in range(len(X) - seq_len):
            self.X.append(X[i:i+seq_len])
            self.y.append(y[i+seq_len])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).view(-1, 1)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ========== 데이터 수집 및 처리 ==========
all_data = []
for ticker in TICKERS:
    print(f"📥 다운로드 중: {ticker}")
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
    df = compute_technical_indicators(df)
    df['Ticker'] = ticker
    df = df.dropna()
    all_data.append(df)

combined_df = pd.concat(all_data).reset_index(drop=True)
combined_df['Ticker_enc'] = LabelEncoder().fit_transform(combined_df['Ticker'])

# ========== 정규화 및 LSTM 학습 ==========
X = combined_df[TECHNICAL_FEATURES + ['Ticker_enc']].values
y = combined_df['Adj Close'].values.reshape(-1, 1)
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

dataset = StockDataset(X_scaled, y_scaled, SEQ_LEN)
train_size = int(len(dataset) * 0.8)
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

model = LSTMModel(input_size=X.shape[1]).to(DEVICE)
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
    print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")

# ========== 예측 및 평가 ==========
model.eval()
preds, actuals = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        out = model(xb.to(DEVICE)).cpu().numpy()
        preds.append(out)
        actuals.append(yb.numpy())

preds = np.vstack(preds)
actuals = np.vstack(actuals)
preds_inv = scaler_y.inverse_transform(preds)
actuals_inv = scaler_y.inverse_transform(actuals)
rmse = np.sqrt(mean_squared_error(actuals_inv, preds_inv))

print(f"\n✅ 통합 데이터셋 RMSE: ${rmse:.2f}")

# 평균 주가 대비 RMSE 비율
mean_price = actuals_inv.mean()
error_pct = (rmse / mean_price) * 100
print(f"📏 평균 주가: ${mean_price:.2f}, RMSE 비율: {error_pct:.2f}%")