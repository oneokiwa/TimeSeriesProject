# 3. xgboost으로 통합 데이터셋에서 기술적 지표 조합별 성능 비교

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

plt.rcParams['font.family'] = ['Malgun Gothic', 'Segoe UI Emoji']
plt.rcParams['axes.unicode_minus'] = False

# ========== 설정 ==========
TICKERS = ['HD', 'MCD', 'LOW', 'TJX', 'SBUX', 'NKE', 'MAR', 'CMG', 'ORLY', 'TGT']
PERIOD = "2y"
INTERVAL = "1d"
TECHNICAL_FEATURES = ['SMA_20', 'EMA_20', 'RSI_14', 'ROC_10', 'Stoch_%K', 'Stoch_%D',
                      'BB_upper', 'BB_lower', 'ATR_14', 'MACD', 'MACD_signal', 'CCI_14', 'OBV']

# ========== 기술적 지표 계산 함수 ==========
def compute_technical_indicators(df):
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

    return df

# ========== 전체 종목 데이터 수집 ==========
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

combined_df = pd.concat(all_data).reset_index()

# ========== 학습용 데이터 준비 ==========
le = LabelEncoder()
combined_df['Ticker_enc'] = le.fit_transform(combined_df['Ticker'])

X = combined_df[TECHNICAL_FEATURES + ['Ticker_enc']]
y = combined_df['Adj Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ========== 모델 학습 ==========
model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ========== 성능 출력 ==========
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mean_price = y.mean()
error_pct = (rmse / mean_price) * 100

print(f"\n📊 평균 수정 종가: ${mean_price:.2f}")
print(f"✅ MSE: {mse:.2f} (달러²)")
print(f"📏 RMSE: ${rmse:.2f} → 평균 대비 {error_pct:.2f}%")

# ========== XGBoost 중요도 시각화 ==========
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.title("📈 XGBoost Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# ========== SHAP 값 분석 ==========
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# SHAP 평균 중요도
shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)
