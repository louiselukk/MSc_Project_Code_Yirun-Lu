import numpy as np

class Market:
    def __init__(self, initial_price=100, long_term_mean=100, spread=1,
                 alpha=0.1, beta=0.05, noise_std=0.1, window_size=10):
        self.traded_volume = 0  # Cumulative volume
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
                kept = [(v, t) for (v, t) in orders if t >= keep_step]
                if kept:
                    new_order_book[price] = kept
            # Update order_book[side]
            self.order_book[side] = new_order_book

         # Calculate the execution price: mid ± spread/2
        execution_price = self.price - self.spread/2 if order_type == 'buy' else self.price + self.spread/2

        # Add this order to the order_book with timestep
        side_book = self.order_book[order_type]
        side_book.setdefault(execution_price, []).append((quantity, self.current_step))
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
        if len(self.price_history) > 1:
            returns = np.diff(self.price_history) / np.array(self.price_history[:-1])
            return float(np.std(returns) * np.sqrt(252))
        return 0.0
