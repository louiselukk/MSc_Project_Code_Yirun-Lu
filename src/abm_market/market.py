class Market:
    def __init__(self, initial_price=100, long_term_mean=100, spread=1, alpha=0.1, beta=0.05, noise_std=0.1, window_size=10):
        self.traded_volume = 0   # Cumulative volume
        self.price = initial_price   # Current market price
        self.long_term_mean = long_term_mean   # Target price for mean reversion
        self.price_history = [initial_price]  # Record updated price
        self.step_price_history = [initial_price]   # Record the price after every time step

        # Parameters
        self.spread = spread          # Fixed bid-ask spread
        self.alpha = alpha            # Impact factor
        self.beta = beta              # Mean-reversion strength
        self.noise_std = noise_std    # Random noise
        self.window_size = window_size   # Size of time window to retain order book history (in steps)

        # Order book: price -> quantity and time step
        self.order_book = {'buy': {}, 'sell': {}}
        self.current_step = 0

    def place_order(self, order_type, quantity):
        keep_step = self.current_step - self.window_size

        # Filter no-need orderbook and replace with the new one
        for side in ['buy', 'sell']:
            new_order_book = {}
            for price, orders in self.order_book[side].items():
                # Keep the transactions which satisfy time step >= keep_step
                kept = [(values, time_step) for (values, time_step) in orders if time_step >= keep_step]
                if kept:
                    new_order_book[price] = kept
            # Update order_book[side]
            self.order_book[side] = new_order_book

        # Calculate the execution price: mid ± spread/2
        if order_type == 'buy':
            execution_price = self.price - self.spread / 2
        else:
            execution_price = self.price + self.spread / 2

        # Add this order to the order_book with timestep
        side_book = self.order_book[order_type]
        if execution_price not in side_book:
            side_book[execution_price] = []
        side_book[execution_price].append((quantity, self.current_step))
        self.traded_volume += quantity

        # A_b: total quantity of buying, A_s: total quantity of selling
        A_b = sum(sum(values for values, time_step in orders) for orders in self.order_book['buy'].values())
        A_s = sum(sum(values for values, time_step in orders) for orders in self.order_book['sell'].values())
        Z = (A_b - A_s) / (A_b + A_s) if (A_b + A_s) > 0 else 0

        # Mean‐reversion: "pure impact" price + beta*(long_term_mean − old_price)
        noise = np.random.normal(0, self.noise_std)
        r = self.alpha * Z + noise
        self.price = self.price * np.exp(r) + self.beta * (self.long_term_mean - self.price)
        #self.price = self.price * np.exp(r)

        self.price_history.append(self.price)
        return execution_price

    def step_advance(self):
        self.current_step += 1
        self.step_price_history.append(self.price)

    def get_volatility(self):
        # volatility = std(returns) * sqrt(252), returns = diff(price_history) / price_history[:-1]
        if len(self.price_history) > 1:
            returns = np.diff(self.price_history) / np.array(self.price_history[:-1])
            return np.std(returns) * np.sqrt(252)  # np.sqrt(252): turn daily volatility to annual volatility, 252: number of trade days
        return 0


class Agent:
    def __init__(self, market, initial_capital=10000):
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
        pass

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
        #  Check if we should switch strategy
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

        # If we don’t have at least two prices yet, we can’t compare
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

        # If price history is not long enough, skip trading
        if len(self.market.price_history) < self.lookback + 1:
            self.profit_history.append(self.calculate_pnl())
            return

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
