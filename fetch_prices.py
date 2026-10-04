# fetch_prices.py
import os
import sys
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

# Configuration constants
DB_NAME = "pricing_data.db"
DEFAULT_TICKERS = ["NVDA", "AMD", "INTC"]
LOOKBACK_YEARS = 5

def initialize_pricing_database():
    """
    Initializes the SQLite database with a dedicated index 
    to handle high-speed pricing retrievals.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create the target tracking layout
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pricing (
            ticker TEXT,
            date TEXT,
            adj_close REAL,
            extracted_at TEXT,
            PRIMARY KEY (ticker, date)
        )
    """)
    
    # Create a compound index for fast ordering and filtering
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_pricing_ticker_date 
        ON pricing (ticker, date ASC);
    """)
    
    conn.commit()
    return conn

def load_tickers_from_file(file_path=None):
    """
    Reads a line-separated text file of stock symbols. 
    Falls back to default sector proxies if no file path is given or found.
    """
    if not file_path or not os.path.exists(file_path):
        if file_path:
            print(f"⚠️ Warning: File '{file_path}' not found. Using default symbols instead.")
        else:
            print("ℹ️ No ticker file specified. Falling back to default baseline universe.")
        return DEFAULT_TICKERS

    tickers = []
    with open(file_path, "r") as f:
        for line in f:
            symbol = line.strip().upper()
            if symbol and not symbol.startswith("#"): # Skip blank spaces and comments
                tickers.append(symbol)
                
    if not tickers:
        print("⚠️ Warning: Specified file was empty. Using defaults.")
        return DEFAULT_TICKERS
        
    return list(set(tickers)) # Return deduplicated array

def main():
    # Detect optional command line argument for a custom ticker file path
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    tickers_to_fetch = load_tickers_from_file(input_file)
    print(f"🚀 Initializing Pricing Pipeline. Target Universe: {tickers_to_fetch}")
    
    conn = initialize_pricing_database()
    
    # Configure exact 5-year tracking boundaries
    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_YEARS * 365)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    for symbol in tickers_to_fetch:
        print(f"🔍 Downloading 5-Year historical pricing context for: {symbol}...")
        try:
            # Download daily tracks using yfinance
            ticker_obj = yf.Ticker(symbol)
            df = ticker_obj.history(start=start_str, end=end_str, interval="1d")
            
            if df.empty:
                print(f"  ⚠️ No historical pricing data returned for symbol: {symbol}")
                continue
                
            # Normalize standard dataframe structures
            df = df.reset_index()
            
            # yfinance returns 'Date' as localized datetime timestamps; standardize to simple string fields
            df['date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            
            # Ensure safe fallback extraction if Adjusted Close formatting varies across library updates
            if 'Adj Close' in df.columns:
                df['adj_close'] = df['Adj Close']
            elif 'Close' in df.columns:
                df['adj_close'] = df['Close']
            else:
                print(f"  ❌ Failed to isolate a clear closure metric tracking array for {symbol}.")
                continue
                
            # Drop empty columns and append meta logs
            df = df.dropna(subset=['adj_close'])
            df['ticker'] = symbol
            df['extracted_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Reorder explicitly to line up against our SQLite primary mapping configuration layout
            df_to_save = df[['ticker', 'date', 'adj_close', 'extracted_at']]
            
            # Stream batch arrays using a high performance UPSERT syntax structure
            cursor = conn.cursor()
            inserted_count = 0
            
            for _, row in df_to_save.iterrows():
                cursor.execute("""
                    INSERT INTO pricing (ticker, date, adj_close, extracted_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(ticker, date) DO UPDATE SET
                        adj_close = excluded.adj_close,
                        extracted_at = excluded.extracted_at
                """, tuple(row))
                inserted_count += 1
                
            conn.commit()
            print(f"  ✅ Successfully loaded {inserted_count} trading entries for {symbol}.")
            
        except Exception as e:
            print(f"  ❌ Failed processing pipeline data sequences for ticker '{symbol}': {e}")
            continue

    conn.close()
    print("\n🏁 Pricing history ingestion sequence finished successfully.")

if __name__ == "__main__":
    main()
