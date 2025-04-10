# 1. 'HD' 종목만 가져와서 테스트 해 본 코드

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Malgun Gothic', 'Segoe UI Emoji']
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 다운로드
ticker = "HD"
## 최근 2년치 데이터
data = yf.download(ticker, period="2y", auto_adjust=False)

# 2. 컬럼이 다중이면 평탄화
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# 3. 기술적 지표 수동 계산
df = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()

# SMA, EMA
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

# RSI
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI_14'] = 100 - (100 / (1 + rs))

# ROC
df['ROC_10'] = df['Close'].pct_change(periods=10) * 100

# Stochastic
low_14 = df['Low'].rolling(window=14).min()
high_14 = df['High'].rolling(window=14).max()
df['Stoch_%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()

# Bollinger Bands
ma20 = df['Close'].rolling(window=20).mean()
std20 = df['Close'].rolling(window=20).std()
df['BB_upper'] = ma20 + 2 * std20
df['BB_lower'] = ma20 - 2 * std20

# ATR
tr = np.maximum(df['High'] - df['Low'],
                np.maximum(abs(df['High'] - df['Close'].shift(1)),
                           abs(df['Low'] - df['Close'].shift(1))))
df['ATR_14'] = tr.rolling(window=14).mean()

# MACD
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# CCI
tp = (df['High'] + df['Low'] + df['Close']) / 3
tp_ma = tp.rolling(window=14).mean()
tp_std = tp.rolling(window=14).std()
df['CCI_14'] = (tp - tp_ma) / (0.015 * tp_std)

# OBV
df['OBV'] = np.where(df['Close'] > df['Close'].shift(1),
                     df['Volume'],
                     np.where(df['Close'] < df['Close'].shift(1),
                              -df['Volume'], 0))
df['OBV'] = df['OBV'].cumsum()

# 4. 모델 학습 데이터 준비
features = ['SMA_20', 'EMA_20', 'RSI_14', 'ROC_10', 'Stoch_%K', 'Stoch_%D',
            'BB_upper', 'BB_lower', 'ATR_14', 'MACD', 'MACD_signal', 'CCI_14', 'OBV']
df = df.dropna()

X = df[features]
y = df['Adj Close']

# 8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 5. XGBoost 모델 학습
model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train)

# 6. 성능 출력
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"✅ Mean Squared Error: {mse:.2f}")

# 7. 중요도 시각화
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\n📊 기술적 지표 중요도 순위:")
print(importance_df)

# 시각화
plt.figure(figsize=(12,8))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title("XGBoost Feature Importance (기술적 지표)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 1. 평균 수정종가 계산
mean_price = df['Adj Close'].mean()

# 2. MSE 및 RMSE 계산
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

# 3. 출력
print(f"\n📊 최근 2년간 평균 수정종가: ${mean_price:.2f}")
print(f"✅ MSE (평균 제곱 오차): {mse:.2f} (단위: 달러²)")
print(f"📏 RMSE (예측 오차의 평균 크기): ${rmse:.2f} (단위: 달러)")

# 4. 예측 성능 해석 문구 출력
error_ratio = (rmse / mean_price) * 100

if error_ratio < 2:
    level = "🎯 매우 정확한 예측입니다!"
elif error_ratio < 5:
    level = "✅ 꽤 정확한 예측입니다."
elif error_ratio < 10:
    level = "⚠️ 어느 정도 차이가 존재합니다."
else:
    level = "❌ 예측 오차가 꽤 큰 편입니다. 개선이 필요합니다."

print(f"📉 평균 주가 대비 RMSE 비율: {error_ratio:.2f}% → {level}")

import shap

# 1. SHAP 값 계산
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# 2. SHAP Summary Plot (특성별 중요도 시각화)
shap.summary_plot(shap_values, X_train)

# 3. SHAP Dependence Plot (특정 피처의 영향력 시각화)
shap.dependence_plot('BB_upper', shap_values, X_train)