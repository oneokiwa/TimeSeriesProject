# 2. xgboost로 피쳐중요도, shap value 맷플롯립 시각화

import shap
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

plt.rcParams['font.family'] = ['Malgun Gothic', 'Segoe UI Emoji']
plt.rcParams['axes.unicode_minus'] = False

tickers = ['HD', 'MCD', 'LOW', 'TJX', 'SBUX', 'NKE', 'MAR', 'CMG', 'ORLY', 'TGT']

# 반복 전: 지표별 중요도 누적 저장용
importance_dict = {feature: [] for feature in [
    'SMA_20', 'EMA_20', 'RSI_14', 'ROC_10', 'Stoch_%K', 'Stoch_%D',
    'BB_upper', 'BB_lower', 'ATR_14', 'MACD', 'MACD_signal', 'CCI_14', 'OBV'
]}

# shap 지표별 중요도 누적 저장용
shap_mean_dict = {feature: [] for feature in [
    'SMA_20', 'EMA_20', 'RSI_14', 'ROC_10', 'Stoch_%K', 'Stoch_%D',
    'BB_upper', 'BB_lower', 'ATR_14', 'MACD', 'MACD_signal', 'CCI_14', 'OBV'
]}

for ticker in tickers:
    print(f"\n\n📈 [{ticker}] 종목 분석 시작")
    try:
        data = yf.download(ticker, period="2y", interval="1d", auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        df = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()

        # 기술적 지표 계산
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

        features = ['SMA_20', 'EMA_20', 'RSI_14', 'ROC_10', 'Stoch_%K', 'Stoch_%D',
                    'BB_upper', 'BB_lower', 'ATR_14', 'MACD', 'MACD_signal', 'CCI_14', 'OBV']
        df = df.dropna()
        X = df[features]
        y = df['Adj Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mean_price = df['Adj Close'].mean()
        error_ratio = (rmse / mean_price) * 100

        # 7. 중요도 시각화
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        print("\n📊 기술적 지표 중요도 순위:")
        print(importance_df)

        # 시각화
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.gca().invert_yaxis()
        plt.title(f"[{ticker}] 기술적 지표 중요도 (XGBoost)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

        # 중요도 저장
        for feat, imp in zip(X.columns, importances):
            importance_dict[feat].append(imp)

        # SHAP 계산 (tree explainer 사용)
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

        # 평균 SHAP 중요도 저장
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)  # 각 feature별 평균 |SHAP|
        for feat, shap_val in zip(X.columns, mean_abs_shap):
            shap_mean_dict[feat].append(shap_val)

        # 분포 시각화
        shap.summary_plot(shap_values, X_test, show=True)  # 샘플별 분포 시각화


        print(f"📊 평균 수정 종가: ${mean_price:.2f}")
        print(f"✅ MSE: {mse:.2f} (달러²)")
        print(f"📏 RMSE: ${rmse:.2f} → 평균 대비 {error_ratio:.2f}%")

        if error_ratio < 2:
            level = "🎯 매우 정확한 예측입니다!"
        elif error_ratio < 5:
            level = "✅ 꽤 정확한 예측입니다."
        elif error_ratio < 10:
            level = "⚠️ 예측 오차가 다소 큽니다."
        else:
            level = "❌ 예측 정확도가 낮습니다. 개선이 필요해요."
        print(f"📈 해석: {level}")

    except Exception as e:
        print(f"❌ {ticker} 분석 중 오류 발생: {e}")

# 평균 중요도 계산
avg_importance_df = pd.DataFrame({
    'Feature': list(importance_dict.keys()),
    'Average Importance': [np.mean(importance_dict[f]) for f in importance_dict]
}).sort_values(by='Average Importance', ascending=False)

# 출력
print("\n📊 [전체 종목 평균] 기술적 지표 중요도:")
print(avg_importance_df)

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(avg_importance_df['Feature'], avg_importance_df['Average Importance'])
plt.gca().invert_yaxis()
plt.title("📊 전체 종목 평균 기술적 지표 중요도")
plt.xlabel("Average Importance")
plt.tight_layout()
plt.show()

# SHAP 평균 중요도 계산
shap_avg_df = pd.DataFrame({
    'Feature': list(shap_mean_dict.keys()),
    'Average SHAP Value': [np.mean(shap_mean_dict[f]) for f in shap_mean_dict]
}).sort_values(by='Average SHAP Value', ascending=False)

# 출력
print("\n📊 [SHAP 기준] 전체 종목 평균 기술적 지표 영향력:")
print(shap_avg_df)

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(shap_avg_df['Feature'], shap_avg_df['Average SHAP Value'], color='coral')
plt.gca().invert_yaxis()
plt.title("🔥 SHAP 기반 기술적 지표 평균 영향력 (전체 종목)")
plt.xlabel("Average |SHAP Value|")
plt.tight_layout()
plt.show()
