# database.py
import sqlite3
import pandas as pd
import numpy as np
import config

def extract_and_pivot_statement(conn, table_name, ticker, metrics_list):
    """
    Extracts individual long relational rows from your yfinance scraper table
    and pivots them back into a wide dataframe with metric columns.
    """
    metrics_placeholder = ",".join([f"'{m}'" for m in metrics_list])
    query = f"""
        SELECT fiscal_date, metric, value 
        FROM {table_name} 
        WHERE ticker = '{ticker}' AND metric IN ({metrics_placeholder})
        ORDER BY fiscal_date ASC
    """
    df = pd.read_sql_query(query, conn, parse_dates=['fiscal_date'])
    
    if df.empty:
        return pd.DataFrame()
        
    df_pivoted = df.pivot(index='fiscal_date', columns='metric', values='value')
    return df_pivoted

def load_financial_data():
    """
    Reads relational statements from trading_results.db, structures KPIs,
    and returns a combined dataset securely aligned with daily pricing tracks.
    """
    # 1. Fetch, parse, and generate pricing returns data
    conn_price = sqlite3.connect(config.PRICING_DB_PATH)
    price_dfs = []
    for ticker in config.TICKERS:
        query = f"SELECT date, adj_close FROM pricing WHERE ticker='{ticker}' ORDER BY date ASC"
        df = pd.read_sql_query(query, conn_price, parse_dates=['date'])
        df = df.set_index('date').rename(columns={'adj_close': ticker})
        price_dfs.append(df)
    conn_price.close()
    
    pricing_matrix = pd.concat(price_dfs, axis=1, join='inner')
    returns_matrix = pricing_matrix.pct_change().dropna()
    aligned_dates = returns_matrix.index

    # 2. Extract specific fundamental components from trading_results.db
    conn_fund = sqlite3.connect(config.FUNDAMENTAL_DB_PATH)
    fund_dfs = []
    
    # Standard yfinance scraped text identifiers mapped dynamically
    inc_metrics = ['Total Revenue', 'Gross Profit']
    bal_metrics = ['Total Assets', 'Total Liabilities Net Minority Interest']
    
    for ticker in config.TICKERS:
        df_inc = extract_and_pivot_statement(conn_fund, "income_statement", ticker, inc_metrics)
        df_bal = extract_and_pivot_statement(conn_fund, "balance_sheet", ticker, bal_metrics)
        
        if df_inc.empty or df_bal.empty:
            raise ValueError(f"Required fundamental indicators missing in database for ticker: {ticker}")
            
        # Join sheets on matching fiscal dates
        df_ticker_fund = pd.concat([df_inc, df_bal], axis=1, join='outer').sort_index()
        
        # Calculate trailing/quarterly structural KPIs dynamically
        df_ticker_fund['gross_margin'] = df_ticker_fund['Gross Profit'] / df_ticker_fund['Total Revenue']
        df_ticker_fund['yoy_growth'] = df_ticker_fund['Total Revenue'].pct_change(periods=4)
        df_ticker_fund['leverage_ratio'] = (
            df_ticker_fund['Total Liabilities Net Minority Interest'] / df_ticker_fund['Total Assets']
        )
        
        kpi_cols = ['gross_margin', 'yoy_growth', 'leverage_ratio']
        df_kpi = df_ticker_fund[kpi_cols].copy()
        df_kpi = df_kpi.rename(columns={col: f"{ticker}_{col}" for col in kpi_cols})
        fund_dfs.append(df_kpi)
        
    conn_fund.close()
    
    # Merge quarterly stock variables together
    fundamentals_matrix = pd.concat(fund_dfs, axis=1, join='outer')
    
    # --- THE FIX: Create a complete time union index containing pricing dates ---
    # This inserts the empty trading days into the index framework without losing quarterly values
    full_date_index = fundamentals_matrix.index.union(aligned_dates)
    fundamentals_matrix = fundamentals_matrix.reindex(full_date_index)
    
    # Now cleanly forward-fill the sparse quarterly metrics through the daily trading gaps
    fundamentals_matrix = fundamentals_matrix.ffill().bfill()
    
    # Slice the daily-aligned fundamentals exactly onto our tracking calendar timeline
    fundamentals_matrix = fundamentals_matrix.loc[aligned_dates]
    
    return fundamentals_matrix, returns_matrix
