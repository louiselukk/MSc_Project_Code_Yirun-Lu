import numpy as np
from .market import Market  

class Agent:
    def __init__(self, market: Market, initial_capital=10000):
        self.market = market
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.capital_history = [initial_capital]
        self.inventory = 0
        self.inventory_history = [0]
        self.profit_history = [0]
        self.strategy_switch_count = 0

    def record_inventory(self):
        self.inventory_history.append(self.inventory)
        self.capital_history.append(self.capital)

    def calculate_pnl(self):
        return self.capital + self.inventory * self.market.price - self.initial_capital

    def evaluate_strategy_switch(self, avg_pnl, current_step, delta_threshold):
        if len(self.profit_history) < 2:
            return
        pnl_diff = self.profit_history[-1] - self.profit_history[-2]
        if pnl_diff < delta_threshold and self.profit_history[-1] < avg_pnl:
            self.switch_strategy(current_step)

    def action(self, avg_pnl, current_step, delta_threshold):
        self.evaluate_strategy_switch(avg_pnl, current_step, delta_threshold)

class MarketMaker(Agent):
    def __init__(self, market, initial_capital, order_size, target_inventory, tolerance):
        super().__init__(market, initial_capital)
        self.order_size = order_size
        self.target_inventory = target_inventory
        self.tolerance = tolerance
        self.strategy = 'passive'  # or 'adaptive'

    def switch_strategy(self, current_step):
        self.strategy = 'adaptive' if self.strategy == 'passive' else 'passive'
        self.strategy_switch_count += 1

    def action(self, avg_pnl, current_step, delta_threshold):
        self.evaluate_strategy_switch(avg_pnl, current_step, delta_threshold)

        diff = self.inventory - self.target_inventory
        if diff < -self.tolerance:
            qty = min(self.order_size, int(-diff))
            price = self.market.place_order('buy', qty)
            self.capital -= price * qty
            self.inventory += qty
        elif diff > self.tolerance:
            qty = min(self.order_size, int(diff))
            price = self.market.place_order('sell', qty)
            self.capital += price * qty
            self.inventory -= qty

        base_size = self.order_size if self.strategy == 'passive' else (0 if self.market.get_volatility() > 0.5 else int(self.order_size * 1.5))
        buy_size  = base_size + max(-diff, 0)
        sell_size = base_size + max(diff, 0)

        if buy_size > 0:
            p = self.market.place_order('buy', buy_size)
            self.capital -= p * buy_size
            self.inventory += buy_size
        if sell_size > 0:
            p = self.market.place_order('sell', sell_size)
            self.capital += p * sell_size
            self.inventory -= sell_size

        self.profit_history.append(self.calculate_pnl())
        self.record_inventory()

class Arbitrageur(Agent):
    def __init__(self, market, initial_capital, order_size, threshold_arb):
        super().__init__(market, initial_capital)
        self.order_size = order_size
        self.threshold_arb = threshold_arb
        self.strategy = 'hf'  # or 'lf'
        self.trade_records = []

    def switch_strategy(self, current_step):
        self.strategy = 'lf' if self.strategy == 'hf' else 'hf'
        self.strategy_switch_count += 1

    def action(self, avg_pnl, current_step, delta_threshold):
        self.evaluate_strategy_switch(avg_pnl, current_step, delta_threshold)
        if len(self.market.price_history) < 2:
            self.profit_history.append(self.calculate_pnl())
            return

        prev_price = self.market.price_history[-2]
        if self.strategy == 'hf':
            if self.market.price > prev_price * (1 + self.threshold_arb):
                p = self.market.place_order('sell', self.order_size); self.capital += p*self.order_size; self.inventory -= self.order_size
                self.trade_records.append({'step': current_step, 'direction': -1, 'price': self.market.price})
            elif self.market.price < prev_price * (1 - self.threshold_arb):
                p = self.market.place_order('buy', self.order_size); self.capital -= p*self.order_size; self.inventory += self.order_size
                self.trade_records.append({'step': current_step, 'direction': +1, 'price': self.market.price})
        else:
            lf_th, lf_size = self.threshold_arb * 1.5, max(1, int(self.order_size / 2))
            if self.market.price > prev_price * (1 + lf_th):
                p = self.market.place_order('sell', lf_size); self.capital += p*lf_size; self.inventory -= lf_size
            elif self.market.price < prev_price * (1 - lf_th):
                p = self.market.place_order('buy', lf_size); self.capital -= p*lf_size; self.inventory += lf_size

        self.profit_history.append(self.calculate_pnl())
        self.record_inventory()

class ShortTermReversalTrader(Agent):
    def __init__(self, market, initial_capital, order_size, lookback):
        super().__init__(market, initial_capital)
        self.order_size = order_size
        self.lookback = lookback
        self.strategy = 'mean'  # or 'momentum'

    def switch_strategy(self, current_step):
        self.strategy = 'momentum' if self.strategy == 'mean' else 'mean'
        self.strategy_switch_count += 1

    def action(self, avg_pnl, current_step, delta_threshold):
        self.evaluate_strategy_switch(avg_pnl, current_step, delta_threshold)
        if len(self.market.price_history) < self.lookback + 1:
            self.profit_history.append(self.calculate_pnl()); return

        if self.strategy == 'mean':
            mean_price = float(np.mean(self.market.price_history[-self.lookback:]))
            if self.market.price < mean_price:
                p = self.market.place_order('buy', self.order_size);  self.capital -= p*self.order_size; self.inventory += self.order_size
            elif self.market.price > mean_price:
                p = self.market.place_order('sell', self.order_size); self.capital += p*self.order_size; self.inventory -= self.order_size
        else:
            diff = self.market.price - self.market.price_history[-self.lookback]
            if diff > 0:
                p = self.market.place_order('sell', self.order_size); self.capital += p*self.order_size; self.inventory -= self.order_size
            elif diff < 0:
                p = self.market.place_order('buy', self.order_size);  self.capital -= p*self.order_size; self.inventory += self.order_size

        self.profit_history.append(self.calculate_pnl())
        self.record_inventory()
