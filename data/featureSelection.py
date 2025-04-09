import yfinance as yf
import pandas as pd
import numpy as np

# 1. 단일 티커로 다운로드
ticker = "HD"
data = yf.download(ticker, period="3000d", auto_adjust=False)

# 2. 단일 레벨 DataFrame 만들기
df = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()

# 3. 기술적 지표 수동 계산
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI_14'] = 100 - (100 / (1 + rs))

df['ROC_10'] = df['Close'].pct_change(periods=10) * 100

low_14 = df['Low'].rolling(window=14).min()
high_14 = df['High'].rolling(window=14).max()
df['Stoch_%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()

ma20 = df['Close'].rolling(window=20).mean()
std20 = df['Close'].rolling(window=20).std()
df['BB_upper'] = ma20 + 2 * std20
df['BB_lower'] = ma20 - 2 * std20

tr = np.maximum(df['High'] - df['Low'],
                np.maximum(abs(df['High'] - df['Close'].shift(1)),
                           abs(df['Low'] - df['Close'].shift(1))))
df['ATR_14'] = tr.rolling(window=14).mean()

ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

tp = (df['High'] + df['Low'] + df['Close']) / 3
tp_ma = tp.rolling(window=14).mean()
tp_std = tp.rolling(window=14).std()
df['CCI_14'] = (tp - tp_ma) / (0.015 * tp_std)

df['OBV'] = np.where(df['Close'] > df['Close'].shift(1),
                     df['Volume'],
                     np.where(df['Close'] < df['Close'].shift(1),
                              -df['Volume'], 0))
df['OBV'] = df['OBV'].cumsum()

# 4. 상관관계 분석
indicators = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
df_clean = df[['Adj Close'] + indicators].dropna()
correlation = df_clean.corr()['Adj Close'].sort_values(ascending=False)

# 5. 출력
correlation_df = correlation.reset_index()
correlation_df.columns = ['Indicator', 'Correlation with Adj Close']
print(correlation_df)
