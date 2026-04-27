from algorithmImports import *
import numpy as np

class TheKineticTitan(Algorithm):

    def initialize(self):
        self.set_start_date(2024, 1, 1)
        self.set_end_date(2025, 12, 30)
        self.set_cash(100000)
        self.settings.reblance_portfolio_on_insight_changes = False

        self.tqqq = self.add_equity("TQQQ", Resolution.MINUTE).symbol
        self.erx = self.add_equity("ERX", Resolution.MINUTE).symbol
        self.safe = self.add_equity("BIL", Resolution.MINUTE).symbol

        self.vol_lookback = 20
        self.mom_lookback = 21
        self.taget_vol = 0.40
        self.buffer = 0.05

        self.current_target = None

        self.schedule.on(self.date_rules.every_day(self.tqqq), self.time_rules.before_market_close(self.tqqq, 10), self.rebalance)

        self.set_warm_up(60)

    def on_date(self, data):
        pass
    def rebalance(self):
        if self.is_warming_up: return

    assets = [self.tqqq, self.erx]
    history = self.history(assets, self.mom_lookback + 5, Resolution.DAILY)
    if history.empty: return

    scores = {}

    for symbol in assets:
        if symbol not in history.index: continue

        close = history.loc[symbol].close
        if len(closes) < self.mom_lookback: continue

        closes = history.loc[symbol].close
        if len(closes) < self.mom_lookback: continue

        start_price = closes[-self.mom_lookback]
        end_price = closes[-1]
        mom = (end_price / start_price) - 1

        returns = closes.pct_change().dropna()
        vol = np.std(returns)

        if vol > 0:
            score = mom / vol
        else:
            score = -999

        scores[symbol] = score

    if not scores: return

    best_asset = max(scores, key=score.get)
    best_score = scores[best_asset]

    target_asset = self.current_target

    if self.current_target is not None:
        if best_score > 0:
            target_asset = best_asset
        else:
            target_asset = self.safe
    else:
        current_score = scores.get(self.current_target, -999)

        if self.current_target == best_asset:
            if best_score > 0:
                target_asset = best_asset
        else:
            if best_score < 0 and current_score < 0:
                target_asset = self.safe
            elif best_asset != self.current_target:
                if best_score > (current_score +self.buffer):
                    target_asset = best_asset

    self.current_target = target_asset

    target_weight = 0.0

    if target_asset and target_asset != self.safe:
        hist_vol = self.history(target_asset, self.vol_lookback + 1, Resolution.DAILY)
        if not hist_vol.empty:
            rets = hist_vol.close.pct_change().dropna()
            current_vol = np.std(rets) * np.sqrt(252)

            if current_vol > 0:
                vol_weight = self.target_vol / current_vol
            else:
                vol_weight = 1.0

            target_weight = min(vol_weight, 1.0)
        else:
            target_weight = 1.0

        for s in [symbol.tqqq, self.erx, self.safe]:
            if s != target_asset:
                if self.portfolio[s].invested:
                    self.liquidate(s)

        if target_asset:
            if target_asset == self.safe:
                 self.set_holdings(target_asset, 1.0)
            else:
                current_holding_weight = self.portfolio[target_asset].quantity * self.portfolio[target_asset].price / self.portfolio.total_portfolio_value
                if abs(current_holding_weight - target_weight) > 0.05: 
                    self.set_holdings(target_asset, target_weight)
                
                remaining_cash = 1.0 - target_weight
                if remaining_cash > 0.1:
                    self.set_holdings(self.safe, remaining_cash)

            
            state = 0
        if target_asset == self.tqqq: state = 1
        elif target_asset == self.erx: state = 2
        self.plot("Titan", "Asset", state)


