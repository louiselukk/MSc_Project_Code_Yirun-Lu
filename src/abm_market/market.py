import numpy as np

class Market:
    def __init__(self, initial_price=100, long_term_mean=100, spread=1,
                 alpha=0.1, beta=0.05, noise_std=0.1, window_size=10):
        self.traded_volume = 0
        self.price = initial_price
        self.long_term_mean = long_term_mean
        self.price_history = [initial_price]
        self.step_price_history = [initial_price]

        self.spread = spread
        self.alpha = alpha
        self.beta = beta
        self.noise_std = noise_std
        self.window_size = window_size

        self.order_book = {'buy': {}, 'sell': {}}
        self.current_step = 0

    def place_order(self, order_type, quantity):
        keep_step = self.current_step - self.window_size

        for side in ['buy', 'sell']:
            new_order_book = {}
            for price, orders in self.order_book[side].items():
                kept = [(v, t) for (v, t) in orders if t >= keep_step]
                if kept:
                    new_order_book[price] = kept
            self.order_book[side] = new_order_book

        execution_price = self.price - self.spread/2 if order_type == 'buy' else self.price + self.spread/2

        side_book = self.order_book[order_type]
        side_book.setdefault(execution_price, []).append((quantity, self.current_step))
        self.traded_volume += quantity

        A_b = sum(sum(v for v, _ in orders) for orders in self.order_book['buy'].values())
        A_s = sum(sum(v for v, _ in orders) for orders in self.order_book['sell'].values())
        Z = (A_b - A_s) / (A_b + A_s) if (A_b + A_s) > 0 else 0.0

        noise = np.random.normal(0, self.noise_std)
        r = self.alpha * Z + noise
        self.price = self.price * np.exp(r) + self.beta * (self.long_term_mean - self.price)
        self.price_history.append(self.price)
        return execution_price

    def step_advance(self):
        self.current_step += 1
        self.step_price_history.append(self.price)

    def get_volatility(self):
        if len(self.price_history) > 1:
            returns = np.diff(self.price_history) / np.array(self.price_history[:-1])
            return float(np.std(returns) * np.sqrt(252))
        return 0.0
