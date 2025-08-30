import numpy as np
from .market import Market  

class Agent:
    def __init__(self, market: Market, initial_capital=10000):
        self.market = market
        self.initial_capital = initial_capital
        self.capital = initial_capital  # Current capital
        self.capital_history = [initial_capital]
        self.inventory = 0  # Current inventory
        self.profit_history = [0]  # P&L history
        self.inventory_history = [0]  # Inventory history
        self.strategy_switch_count = 0  # Number of times of switching strategies

    def record_inventory(self):
        # Record the inventory and capital after each action
        self.inventory_history.append(self.inventory)
        self.capital_history.append(self.capital)

    def calculate_pnl(self):
        # P&L = capital + inventory * price – initial capital
        return self.capital + self.inventory * self.market.price - self.initial_capital
        
    def evaluate_strategy_switch(self, avg_pnl, current_step, delta_threshold):
         # If current P&L - last P&L < delta_threshold and < avg_pnl，then change strategy
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
        self.strategy = 'passive'  # Either 'passive' or 'adaptive'
        self.target_inventory = target_inventory  # Desired inventory level
        self.tolerance = tolerance

    def switch_strategy(self, current_step):
        self.strategy = 'adaptive' if self.strategy == 'passive' else 'passive'
        self.strategy_switch_count += 1

    def action(self, avg_pnl, current_step, delta_threshold):
        self.evaluate_strategy_switch(avg_pnl, current_step, delta_threshold)

        # Rebalance inventory if we’re too far from target_inventory
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
        # Determine “base_size” depending on strategy
        if self.strategy == 'passive':
            base_size = self.order_size
        else:
            # In adaptive mode, scale base_size by volatility: if vol > self.sigma_th， stop quoting; else increase by 1.5×
            vol = self.market.get_volatility()
            base_size = 0 if vol > 0.5 else int(self.order_size * 1.5)

        # Compute how far inventory is from target, add/subtract from base_size
        buy_size = base_size + max(-diff, 0)
        sell_size = base_size + max(diff, 0)

        # Place the buy order (if buy_size > 0)
        if buy_size > 0:
            price_b = self.market.place_order('buy', buy_size)
            self.capital -= price_b * buy_size
            self.inventory += buy_size

        # Place the sell order (if sell_size > 0)
        if sell_size > 0:
            price_s = self.market.place_order('sell', sell_size)
            self.capital += price_s * sell_size
            self.inventory -= sell_size

        # Update P&L and record inventory/capital history
        self.profit_history.append(self.calculate_pnl())
        self.record_inventory()

class Arbitrageur(Agent):
    def __init__(self, market, initial_capital, order_size, threshold_arb):
        super().__init__(market, initial_capital)
        self.order_size = order_size
        self.threshold_arb = threshold_arb    # Percent price‐move threshold
        self.strategy = 'hf'   # 'high‐frequency' or 'low‐frequency'
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
            # High‐frequency arbitrage: if price > prev * (1 + threshold_arb), sell. If < prev*(1 − threshold_arb), buy.
            if self.market.price > prev_price * (1 + self.threshold_arb):
                price = self.market.place_order('sell', self.order_size)
                self.capital += price * self.order_size
                self.inventory -= self.order_size

                self.trade_records.append({
                    'step': current_step,
                    'direction': -1,
                    'price': self.market.price
                })

            elif self.market.price < prev_price * (1 - self.threshold_arb):
                price = self.market.place_order('buy', self.order_size)
                self.capital -= price * self.order_size
                self.inventory += self.order_size

                self.trade_records.append({
                    'step': current_step,
                    'direction': 1,
                    'price': self.market.price
                })
        else:
            # Low‐frequency arbitrage: threshold_arb is 1.5× bigger, size is halved
            lf_th = self.threshold_arb * 1.5
            lf_size = max(1, int(self.order_size / 2))
            if self.market.price > prev_price * (1 + lf_th):
                price = self.market.place_order('sell', lf_size)
                self.capital += price * lf_size
                self.inventory -= lf_size

                self.trade_records.append({
                    'step': current_step,
                    'direction': -1,
                    'price': self.market.price
                })


            elif self.market.price < prev_price * (1 - lf_th):
                price = self.market.place_order('buy', lf_size)
                self.capital -= price * lf_size
                self.inventory += lf_size

                self.trade_records.append({
                    'step': current_step,
                    'direction': 1,
                    'price': self.market.price
                })

        self.profit_history.append(self.calculate_pnl())
        self.record_inventory()

class ShortTermReversalTrader(Agent):
    def __init__(self, market, initial_capital, order_size, lookback):
        super().__init__(market, initial_capital)
        self.order_size = order_size
        self.lookback = lookback  # # How many past-time steps to reference
        self.strategy = 'mean'  # 'mean' or 'momentum'

    def switch_strategy(self, current_step):
        self.strategy = 'momentum' if self.strategy == 'mean' else 'mean'
        self.strategy_switch_count += 1

    def action(self, avg_pnl, current_step, delta_threshold):
        self.evaluate_strategy_switch(avg_pnl, current_step, delta_threshold)
        if len(self.market.price_history) < self.lookback + 1:
            self.profit_history.append(self.calculate_pnl()); return

        if self.strategy == 'mean':
            # Mean‐reversion: compute average of last `lookback` prices
            recent = self.market.price_history[-self.lookback:]
            mean_price = np.mean(recent)
            if self.market.price < mean_price:
                # If current price < mean of recent, buy
                price = self.market.place_order('buy', self.order_size)
                self.capital -= price * self.order_size
                self.inventory += self.order_size
            elif self.market.price > mean_price:
                # If current price > mean of recent, sell
                price = self.market.place_order('sell', self.order_size)
                self.capital += price * self.order_size
                self.inventory -= self.order_size

        else:
            # Momentum‐reversal: compare current price vs price `lookback` steps ago
            diff = self.market.price - self.market.price_history[-self.lookback]
            if diff > 0:
                # If price has risen over the past lookback steps, sell
                price = self.market.place_order('sell', self.order_size)
                self.capital += price * self.order_size
                self.inventory -= self.order_size
            elif diff < 0:
                # If price has fallen, buy
                price = self.market.place_order('buy', self.order_size)
                self.capital -= price * self.order_size
                self.inventory += self.order_size

        # Update P&L and record inventory/capital
        self.profit_history.append(self.calculate_pnl())
        self.record_inventory()
