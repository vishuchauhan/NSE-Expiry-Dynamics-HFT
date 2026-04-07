import math

class KellyRiskManager:
    def __init__(self, max_allocation=0.20, kelly_fraction=0.5):
        """
        Initializes the Risk Engine.
        max_allocation : Hard cap. Never risk more than 20% of total capital on one trade.
        kelly_fraction : We use 'Half-Kelly' (0.5) to protect against volatility drag.
        """
        self.max_allocation = max_allocation
        self.kelly_fraction = kelly_fraction
        
        # Approximate margin required per NIFTY option selling lot (in INR)
        self.margin_per_lot = 50000 

    def calculate_position_size(self, current_capital, win_prob, avg_win, avg_loss):
        """
        Calculates the exact number of lots to trade based on the Kelly Criterion.
        win_prob : Provided live by the HMM Brain
        avg_win / avg_loss : Historical performance for the current market regime
        """
        # 1. Protect against division by zero or bad data
        if avg_loss == 0 or win_prob <= 0:
            return 0, 0.0

        # 2. Calculate 'b' (The Win/Loss Ratio)
        b = abs(avg_win / avg_loss)
        
        p = win_prob
        q = 1.0 - p
        
        # 3. The Full Kelly Formula: f* = p - (q / b)
        full_kelly = p - (q / b)
        
        # 4. Filter out Negative Expected Value
        if full_kelly <= 0:
            return 0, 0.0 # The math says this trade is a guaranteed long-term loser. Do not trade.
            
        # 5. Apply Fractional Kelly (Institutional Safety Feature)
        adjusted_kelly = full_kelly * self.kelly_fraction
        
        # 6. Apply Hard Capital Cap
        final_allocation_pct = min(adjusted_kelly, self.max_allocation)
        
        # 7. Convert Percentage to Actual Lots
        allocated_capital = current_capital * final_allocation_pct
        num_lots = math.floor(allocated_capital / self.margin_per_lot)
        
        return num_lots, final_allocation_pct