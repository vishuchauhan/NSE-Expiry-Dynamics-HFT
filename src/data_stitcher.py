import pandas as pd
import os
import glob

def stitch_multiple_days(input_folder, output_filename):
    print(f"Scanning folder: {input_folder} for CSV files...")
    
    # Grab all files matching your uploaded naming convention
    all_files = glob.glob(os.path.join(input_folder, "NSE_FNO_DATA_*.csv"))
    
    if not all_files:
        print("No CSV files found! Check your folder path.")
        return

    print(f"Found {len(all_files)} files. Stitching...")
    
    df_list = []
    for file in all_files:
        print(f"Loading {os.path.basename(file)}...")
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)
        
    master_df = pd.concat(df_list, ignore_index=True)
    
    print("Parsing and sorting by time chronologically...")
    try:
        # Crucial: dayfirst=True handles the DD/MM/YYYY format in your CSVs
        master_df['Datetime'] = pd.to_datetime(master_df['Date'].astype(str) + ' ' + master_df['Time'].astype(str), dayfirst=True)
        master_df = master_df.sort_values(by='Datetime')
    except Exception as e:
        print(f"Warning on datetime sort: {e}")

    # Drop duplicates just in case there's overlap
    master_df = master_df.drop_duplicates(subset=['Ticker', 'Datetime'])
    
    output_path = os.path.join(input_folder, output_filename)
    master_df.to_csv(output_path, index=False)
    print(f"\nStitching complete! Saved massive multi-day dataset ({len(master_df)} rows) to:")
    print(output_path)

if __name__ == "__main__":
    # Ensure this points to the folder containing your 5 CSVs
    raw_data_folder = 'data/raw/' 
    stitch_multiple_days(raw_data_folder, 'NIFTY_MULTI_DAY_MASTER.csv')