import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from deap import base, creator, tools, algorithms
import random

# ----------------------------
# 1. 종목 및 데이터 수집
# ----------------------------
tickers = ['AAPL']

def fetch_data():
    frames = []
    for ticker in tickers:
        df = yf.download(ticker, period="3y", interval="1d")
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # ✅ 필요한 컬럼만
        df['Ticker'] = ticker
        frames.append(df)
    all_data = pd.concat(frames)
    all_data.reset_index(inplace=True)
    return all_data

data = fetch_data()

# ----------------------------
# 2. 기술적 지표 추가 함수
# ----------------------------
def add_indicators(df, bb_len=20, bb_std=2, sma_len=20, ema_len=20, wma_len=20,
                   macd_fast=12, macd_slow=26, macd_signal=9, stc_cycle=10,
                   rsi_len=14, roc_len=10):
    df = df.copy()
    try:
        df['SMA'] = ta.sma(df['Close'], length=sma_len)
        df['EMA'] = ta.ema(df['Close'], length=ema_len)
        df['WMA'] = ta.wma(df['Close'], length=wma_len)
        df['RSI'] = ta.rsi(df['Close'], length=rsi_len)
        df['STC'] = ta.stc(df['Close'], cycle=stc_cycle)
        df['ROC'] = ta.roc(df['Close'], length=roc_len)

        bb = ta.bbands(df['Close'], length=bb_len, std=bb_std)
        if isinstance(bb, pd.DataFrame):
            df = pd.concat([df, bb], axis=1)

        macd = ta.macd(df['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
        if isinstance(macd, pd.DataFrame):
            df['MACD'] = macd.iloc[:, 0]
            df['MACD_signal'] = macd.iloc[:, 1]
    except Exception as e:
        print("Indicator error:", e)
    return df

# ----------------------------
# 3. 매매 시그널 생성 함수
# ----------------------------
def make_signal(df):
    required_cols = ['RSI', 'STC', 'BBL_20_2.0', 'BBU_20_2.0', 'MACD', 'MACD_signal']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    available_cols = [col for col in required_cols if col in df.columns and not df[col].isnull().all()]
    if not available_cols:
        df['signal'] = 0
        return df['signal']
    try:
        df = df.dropna(subset=available_cols).copy()
    except:
        df['signal'] = 0
        return df['signal']
    cond_buy = (
        (df['RSI'] < 30) &
        (df['MACD'] > df['MACD_signal']) &
        (df['Close'] < df['BBL_20_2.0']) &
        (df['STC'] < 25)
    )
    cond_sell = (
        (df['RSI'] > 70) &
        (df['MACD'] < df['MACD_signal']) &
        (df['Close'] > df['BBU_20_2.0']) &
        (df['STC'] > 75)
    )
    df['signal'] = np.where(cond_buy, 1, np.where(cond_sell, -1, 0))
    return df['signal']

# ----------------------------
# 4. 평가 함수 (수익률 - MDD 패널티 기반)
# ----------------------------
def evaluate(params):
    bb_len, bb_std, sma_len, ema_len, wma_len, macd_fast, macd_slow, macd_signal, stc_cycle, rsi_len, roc_len = map(int, params)
    total_return = 0
    drawdowns = []
    for ticker in tickers:
        df = data[data['Ticker'] == ticker].copy()
        df = add_indicators(df, bb_len, bb_std, sma_len, ema_len, wma_len,
                            macd_fast, macd_slow, macd_signal, stc_cycle, rsi_len, roc_len)
        df['signal'] = make_signal(df)

        # ✅ 오류 없이 처리되도록 Series 확실하게 보장
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        df['future_return'] = close.pct_change(fill_method=None).shift(-1)

        df['strategy_return'] = df['signal'].shift(1) * df['future_return']
        df['cum_return'] = (1 + df['strategy_return'].fillna(0)).cumprod()
        total_return += df['strategy_return'].sum()

        peak = df['cum_return'].cummax()
        dd = (df['cum_return'] - peak) / peak
        drawdowns.append(dd.min())
    avg_dd = np.mean(drawdowns)
    score = total_return - abs(avg_dd) * 10
    return (score,)

# ----------------------------
# 5. DEAP GA 설정
# ----------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 5, 30)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 11)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# ----------------------------
# 6. 유전 알고리즘 실행
# ----------------------------
print("🚀 유전 알고리즘 실행 중...")
pop = toolbox.population(n=20)
pop, _ = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=5, verbose=True)
best_ind = tools.selBest(pop, 1)[0]
print("✅ 최적 파라미터:", best_ind)

# ----------------------------
# 7. 최종 시그널 생성 및 저장 (지표별 signal 판단) - dict 방식
# ----------------------------
for ticker in tickers:
    df = data[data['Ticker'] == ticker].copy()
    df = add_indicators(df, *best_ind)
    df.dropna(inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    dates = df['Date'].dt.strftime("%Y-%m-%d").tolist()

    # ✅ dict 방식으로 우선 데이터 저장
    signals = {
        'Price': df['Close'].values,
        'SMA': np.where(df['Close'] > df['SMA'], 1, np.where(df['Close'] < df['SMA'], -1, 0)),
        'EMA': np.where(df['Close'] > df['EMA'], 1, np.where(df['Close'] < df['EMA'], -1, 0)),
        'WMA': np.where(df['Close'] > df['WMA'], 1, np.where(df['Close'] < df['WMA'], -1, 0)),
        'BB_lower': np.where(df['Close'] < df['BBL_20_2.0'], 1, 0),
        'BB_upper': np.where(df['Close'] > df['BBU_20_2.0'], -1, 0),
        'MACD': np.where(df['MACD'] > df['MACD_signal'], 1,
                         np.where(df['MACD'] < df['MACD_signal'], -1, 0)),
        'STC': np.where(df['STC'] < 25, 1,
                        np.where(df['STC'] > 75, -1, 0)),
        'RSI': np.where(df['RSI'] < 30, 1,
                        np.where(df['RSI'] > 70, -1, 0)),
        'ROC': np.where(df['ROC'].diff() > 0, 1,
                        np.where(df['ROC'].diff() < 0, -1, 0))
    }

    # ✅ DataFrame으로 변환
    signal_matrix = pd.DataFrame.from_dict(signals, orient='index', columns=dates)

    # ✅ 저장
    signal_matrix.to_csv("signal_matrix_from_GA.csv", index=True, header=True)
    print("📁 signal_matrix_from_GA.csv 저장 완료 (지표별 판단 signal 매트릭스)")
