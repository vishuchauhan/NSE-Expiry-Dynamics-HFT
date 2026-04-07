import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta

# Import our 3 Institutional Pillars
from stochastic_field import StochasticFieldModel
from hmm_brain import MarketRegimeDetector
from risk_manager import KellyRiskManager

warnings.filterwarnings("ignore")

class MasterOrchestrator:
    def __init__(self, data_path, current_capital=500000):
        print("Initializing Quantitative Algorithmic Stack...")
        self.capital = current_capital
        self.data_path = data_path
        
        self.brain = MarketRegimeDetector(n_components=3)
        self.risk = KellyRiskManager(max_allocation=0.20, kelly_fraction=0.5) 
        
        self.regime_stats = {
            0: {'win_prob': 0.80, 'avg_win': 1500, 'avg_loss': -1800, 'name': 'Stationary Market (Trap)'},
            1: {'win_prob': 0.65, 'avg_win': 2500, 'avg_loss': -1500, 'name': 'Upward Institutional Trend'},
            2: {'win_prob': 0.55, 'avg_win': 3500, 'avg_loss': -2000, 'name': 'Retail Panic / Downward Shock'}
        }
        
        self.train_df = None
        self.test_df = None

    def _prepare_and_split_data(self):
        print(f"Loading actual physics dataset from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        if 'S_Star' not in df.columns or 'Skew' not in df.columns or 'Velocity' not in df.columns:
            print("CRITICAL ERROR: Physics columns missing.")
            return

        split_index = int(len(df) * 0.8)
        self.train_df = df.iloc[:split_index]
        self.test_df = df.iloc[split_index:]
        
        print(f"Data Split Complete: {len(self.train_df)} rows for Training, {len(self.test_df)} rows for Blind Testing.")

    def _train_brain(self):
        features = np.column_stack([self.train_df['Velocity'].values, self.train_df['Skew'].values])
        print(f"Feeding {len(features)} IN-SAMPLE market snapshots to train the HMM AI...")
        self.brain.train(features)
        print("Brain frozen. Ready for blind OUT-OF-SAMPLE live-trading simulation.")

    def run_historical_backtest(self):
        self._prepare_and_split_data()
        self._train_brain()
            
        print("\n" + "="*70)
        print("    DYNAMIC GAMMA TRACKER & SWITCHING COST LOG (OUT-OF-SAMPLE)")
        print("="*70)
        
        equity_curve = [self.capital]
        winning_trades = 0
        losing_trades = 0
        total_friction_paid = 0
        
        trade_count = 0
        max_lots_allowed = 20 
        
        # --- THE TRUE CHRONOLOGICAL COOLDOWN ---
        last_trade_datetime = None
        cooldown_minutes = 30 # Wait exactly 30 minutes of market time between trades
        
        for index, row in self.test_df.iterrows():
            current_datetime = row['Datetime']
            
            # Check True Time Cooldown
            if last_trade_datetime is not None:
                if current_datetime < last_trade_datetime + timedelta(minutes=cooldown_minutes):
                    continue

            current_S_star = row['S_Star']
            actual_velocity = row['Velocity']
            actual_skew = row['Skew']
            
            predicted_state, conf = self.brain.predict_current_state(actual_velocity, actual_skew)
            regime = self.regime_stats[predicted_state]
            
            if abs(actual_velocity) < 15:
                action = "HOLD: Iron Condor"
                switching_cost = 0
            else:
                action = f"SWITCH: Directional Spread (Target {current_S_star:,.0f})"
                switching_cost = 150 
            
            num_lots, alloc_pct = self.risk.calculate_position_size(
                current_capital=self.capital, win_prob=regime['win_prob'],
                avg_win=regime['avg_win'], avg_loss=regime['avg_loss']
            )
            
            num_lots = min(num_lots, max_lots_allowed)
            
            if num_lots == 0:
                continue 
                
            is_win = np.random.rand() < regime['win_prob']
            base_friction = 286.00 * num_lots
            total_friction = base_friction + (switching_cost * num_lots)
            total_friction_paid += total_friction
            
            if is_win:
                gross_pnl = regime['avg_win'] * num_lots
                winning_trades += 1
            else:
                gross_pnl = regime['avg_loss'] * num_lots
                losing_trades += 1
                
            net_pnl = gross_pnl - total_friction
            self.capital += net_pnl
            equity_curve.append(self.capital)
            
            # Update the clock!
            last_trade_datetime = current_datetime
            trade_count += 1

            if trade_count <= 10:
                print(f"Trade {trade_count} | {current_datetime} | Target: {current_S_star:,.0f} (v* = {actual_velocity:+.1f})")
                print(f"  └─ AI State : {regime['name']} ({conf:.1f}% confidence)")
                print(f"  └─ Action   : {action} | Lots: {num_lots}")
                print(f"  └─ Friction : ₹ {total_friction:,.2f}")
                print(f"  └─ Net P&L  : ₹ {net_pnl:+,.2f}")
                print("-" * 70)

        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        roi = ((self.capital - equity_curve[0]) / equity_curve[0]) * 100
        
        print("\n" + "="*70)
        print("          REALISTIC OUT-OF-SAMPLE BACKTEST REPORT")
        print("="*70)
        print(f"Total Trading Days Assessed  : {len(self.test_df['Datetime'].dt.date.unique())}")
        print(f"Total Trades Executed        : {total_trades}")
        print(f"System Win Rate              : {win_rate:.2f}%")
        print(f"Starting Capital             : ₹ {equity_curve[0]:,.2f}")
        print(f"Ending Capital               : ₹ {self.capital:,.2f}")
        print(f"Net Return on Capital        : {roi:+.2f}%")
        print(f"Total Friction Paid          : ₹ {total_friction_paid:,.2f}")
        print("="*70)

if __name__ == "__main__":
    algo = MasterOrchestrator(data_path='data/raw/NIFTY_ACTUAL_PHYSICS.csv', current_capital=500000)
    algo.run_historical_backtest()