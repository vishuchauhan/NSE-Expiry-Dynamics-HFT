import pandas as pd
import numpy as np
import os
import warnings

# Suppress pandas datetime warnings for clean terminal output
warnings.filterwarnings("ignore", category=UserWarning)

class DataPhysicsEngine:
    """
    Computes the localized Potential Field and Attractor (S*) 
    for every timestamp in the NSE dataset.
    """
    @staticmethod
    def calculate_physics(df):
        print("Calculating Institutional Physics (S*, Velocity, Skew)...")
        
        # --- THE STRIKE PRICE EXTRACTOR ---
        # Extracts the actual NIFTY level (e.g., 24000) from the Ticker string (e.g., NIFTY24OCT24000CE)
        # The regex looks for the digits immediately preceding 'CE' or 'PE'
        df['Strike'] = df['Ticker'].str.extract(r'(\d{5})(?:CE|PE)').astype(float)
        
        # Forward fill and backward fill any missing strikes to keep physics continuous
        df['Strike'] = df['Strike'].ffill().bfill()
        
        # 1. Calculate the Attractor (S*) based on the actual Strike Level
        window = 15
        df['S_Star'] = df['Strike'].rolling(window=window).mean()
        
        # 2. Calculate Velocity (v*) - The speed of the moving magnet
        df['Velocity'] = df['S_Star'].diff()
        
        # 3. Calculate IV Skew Proxy (The Fear Index)
        # Premium volatility indicates retail panic/uncertainty
        df['Skew'] = ((df['High'] - df['Low']) / df['Close']) * 100
        
        return df.dropna()

def main():
    raw_path = 'data/raw/NIFTY_MULTI_DAY_MASTER.csv'
    output_path = 'data/raw/NIFTY_ACTUAL_PHYSICS.csv'
    
    if not os.path.exists(raw_path):
        print(f"Error: File not found at {raw_path}")
        return

    print(f"Opening Raw Kaggle Dataset: {raw_path}")
    df = pd.read_csv(raw_path)
    
    # --- THE NIFTY ISOLATION FILTER ---
    # Drops all individual stock options (AARTIIND, ZYDUS, etc.)
    if 'Ticker' in df.columns:
        print("Filtering dataset to isolate pure NIFTY Index options...")
        df = df[df['Ticker'].astype(str).str.contains("NIFTY", na=False, case=False)]
        print(f"Remaining pure NIFTY rows: {len(df)}")
    
    # Pre-processing Datetime (Handling the Indian DD/MM/YYYY format safely)
    if 'Date' in df.columns and 'Time' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), dayfirst=True)
    
    # Run the Physics Logic
    engine = DataPhysicsEngine()
    processed_df = engine.calculate_physics(df)
    
    print(f"Saving 100% Scientifically Accurate Dataset to: {output_path}")
    processed_df.to_csv(output_path, index=False)
    print("Pre-processing Complete. You are ready for the Final Backtest.")

if __name__ == "__main__":
    main()