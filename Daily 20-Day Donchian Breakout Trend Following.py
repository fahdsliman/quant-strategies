import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def donchian_backtest(
    ticker="XOM",
    start="2019-01-01",
    end=None,
    lookback=20,
    allow_short=False,
    initial_cash=100_000
):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError("No data returned. Check ticker/date range.")

    # Donchian channel based on prior N days (shifted to avoid lookahead)
    df["upper"] = df["High"].rolling(lookback).max().shift(1)
    df["lower"] = df["Low"].rolling(lookback).min().shift(1)
    df["middle"] = (df["upper"] + df["lower"]) / 2.0

    # Signals (1 = long, -1 = short, 0 = flat)
    df["pos"] = 0

    # Entry/exit rules
    # Enter long if Close > upper
    # Exit long if Close < middle (your rule)
    long_entry = df["Close"] > df["upper"]
    long_exit  = df["Close"] < df["middle"]

    # Optional short logic (similar style)
    short_entry = (df["Close"] < df["lower"]) if allow_short else False
    short_exit  = (df["Close"] > df["middle"]) if allow_short else False

    position = 0
    positions = []

    for i in range(len(df)):
        if np.isnan(df["upper"].iat[i]) or np.isnan(df["lower"].iat[i]):
            positions.append(0)
            continue

        c = df["Close"].iat[i]
        up = df["upper"].iat[i]
        mid = df["middle"].iat[i]
        low = df["lower"].iat[i]

        # decision order similar to QC logic
        if position <= 0 and c > up:
            position = 1
        elif position >= 0 and allow_short and c < low:
            position = -1
        elif position == 1 and c < mid:
            position = 0
        elif position == -1 and c > mid:
            position = 0

        positions.append(position)

    df["pos"] = positions

    # Strategy returns (use yesterday's position on today's return)
    df["ret"] = df["Close"].pct_change().fillna(0.0)
    df["strat_ret"] = df["pos"].shift(1).fillna(0.0) * df["ret"]

    df["equity"] = initial_cash * (1.0 + df["strat_ret"]).cumprod()
    df["buy_hold"] = initial_cash * (1.0 + df["ret"]).cumprod()

    # Basic metrics
    total_return = df["equity"].iloc[-1] / initial_cash - 1
    daily = df["strat_ret"]
    sharpe = np.sqrt(252) * daily.mean() / (daily.std() + 1e-12)
    max_dd = (df["equity"] / df["equity"].cummax() - 1).min()

    print(f"{ticker} | lookback={lookback} | short={allow_short}")
    print(f"Total return: {total_return:.2%}")
    print(f"Sharpe (naive): {sharpe:.2f}")
    print(f"Max drawdown: {max_dd:.2%}")

    # Plot
    plt.figure()
    plt.plot(df.index, df["equity"], label="Strategy")
    plt.plot(df.index, df["buy_hold"], label="Buy & Hold")
    plt.legend()
    plt.title(f"{ticker} Donchian Breakout Backtest")
    plt.show()

    return df

# Example
if __name__ == "__main__":
    donchian_backtest(ticker="XOM", lookback=20, allow_short=False)

