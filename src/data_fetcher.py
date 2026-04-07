import pandas as pd
import re
import warnings

# Suppress minor pandas warnings for clean output
warnings.filterwarnings('ignore')

class NSEDataParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = None
        self.processed_data = None

    def load_and_parse(self):
        """Loads the raw CSV and parses the Ticker strings."""
        print(f"Loading data from {self.filepath}...")
        df = pd.read_csv(self.filepath)
        
        print("Parsing Ticker strings (this may take a few seconds)...")
        # Regex pattern to extract: Underlying, Expiry, Strike, Type
        # Example: NIFTY31OCT2424500CE.NFO
        pattern = r"^([A-Z]+)(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)\.NFO$"
        
        # Extract components into new columns
        extracted = df['Ticker'].str.extract(pattern)
        extracted.columns = ['Underlying', 'Expiry', 'Strike', 'Option_Type']
        
        # Convert Strike to numeric
        extracted['Strike'] = pd.to_numeric(extracted['Strike'])
        
        # Combine back with the main dataframe
        df = pd.concat([df, extracted], axis=1)
        
        # Create a proper Datetime index for time-series analysis
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d-%m-%Y %H:%M:%S')
        df.drop(columns=['Date', 'Time'], inplace=True)
        
        # Drop rows where the regex failed (e.g., futures contracts if any exist)
        df.dropna(subset=['Underlying'], inplace=True)
        
        self.processed_data = df
        print("Parsing complete.")
        return self.processed_data

    def get_expiry_chain(self, underlying, expiry_date):
        """
        Filters the massive dataset down to a single underlying and expiry date.
        """
        if self.processed_data is None:
            raise ValueError("Data not loaded. Run load_and_parse() first.")
            
        filtered_df = self.processed_data[
            (self.processed_data['Underlying'] == underlying) & 
            (self.processed_data['Expiry'] == expiry_date)
        ]
        
        # Sort chronologically and by strike
        filtered_df = filtered_df.sort_values(by=['Datetime', 'Strike'])
        return filtered_df