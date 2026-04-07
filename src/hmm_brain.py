import numpy as np
import pandas as pd
from hmmlearn import hmm
import warnings

# Suppress hmmlearn warnings for cleaner terminal output
warnings.filterwarnings("ignore")

class MarketRegimeDetector:
    def __init__(self, n_components=3, strike_interval=50):
        """
        Initializes the Hidden Markov Model.
        n_components = 3 represents our target market states:
        0: Stationary/Trap, 1: Upward Institutional Trend, 2: Downward Panic
        """
        self.n_components = n_components
        self.strike_interval = strike_interval
        # We use a Gaussian HMM because our features (velocity, skew) are continuous variables
        self.model = hmm.GaussianHMM(n_components=self.n_components, 
                                     covariance_type="full", 
                                     n_iter=1000, 
                                     random_state=42)
        self.is_trained = False

    def calculate_iv_skew_proxy(self, chain, spot_price, distance=200):
        """
        Calculates a proxy for IV Skew (Retail Fear/Greed).
        Instead of solving complex Black-Scholes implied volatilities, 
        we compare the raw premium of Out-of-The-Money (OTM) Calls vs Puts.
        """
        call_strike = round((spot_price + distance) / self.strike_interval) * self.strike_interval
        put_strike = round((spot_price - distance) / self.strike_interval) * self.strike_interval
        
        try:
            call_premium = chain[(chain['Strike'] == call_strike) & (chain['Option_Type'] == 'CE')].iloc[0]['Close']
            put_premium = chain[(chain['Strike'] == put_strike) & (chain['Option_Type'] == 'PE')].iloc[0]['Close']
            
            # Skew > 0 means Puts are more expensive (Fear). Skew < 0 means Calls are more expensive (Greed).
            skew = put_premium - call_premium
            return skew
        except IndexError:
            return 0.0 # Return neutral if strikes are missing

    def prepare_training_features(self, df_history):
        """
        Transforms raw historical data into the feature matrix required by the HMM.
        df_history must contain: ['Datetime', 'Spot', 'S_Star']
        """
        df = df_history.copy()
        
        # 1. Calculate Attractor Velocity (v*)
        # Difference in S* over a rolling 3-period (e.g., 15-minute) window
        df['S_Star_Prev'] = df['S_Star'].shift(3)
        df['Velocity'] = df['S_Star'] - df['S_Star_Prev']
        
        # 2. Add the Skew feature (Assuming you applied calculate_iv_skew_proxy in your main loop)
        # If 'Skew' column doesn't exist yet, we mock it with zeros temporarily for structure
        if 'Skew' not in df.columns:
            df['Skew'] = 0.0 
            
        # Drop NaNs created by shifting
        df = df.dropna(subset=['Velocity', 'Skew'])
        
        # Format for hmmlearn: a 2D numpy array where each row is [Velocity, Skew]
        features = np.column_stack([df['Velocity'].values, df['Skew'].values])
        return features

    def train(self, features):
        """
        Feeds the historical features into the SDE to let the AI discover the 3 regimes.
        """
        print(f"Training Hidden Markov Model on {len(features)} sequential snapshots...")
        self.model.fit(features)
        self.is_trained = True
        print("HMM Training Complete. Regimes Extracted.")

    def predict_current_state(self, current_velocity, current_skew):
        """
        Live Execution Method: Feeds the current 5-minute data point into the trained 
        brain to instantly classify the current market regime.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained on historical data before live prediction.")
            
        current_observation = np.array([[current_velocity, current_skew]])
        state = self.model.predict(current_observation)[0]
        
        # Get the probability confidence of this prediction
        state_probs = self.model.predict_proba(current_observation)[0]
        confidence = state_probs[state] * 100
        
        return state, confidence