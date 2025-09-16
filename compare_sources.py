#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import vnstock as vn

symbol = 'SHB'
print(f'Comparing data sources for {symbol}...')

print("=" * 50)
print("1. Direct vnstock API (TCBS):")
try:
    vs = vn.Vnstock()
    px1 = vs.stock(symbol, 'TCBS').quote.history(start='2024-12-01', end='2024-12-31', interval='1D')
    
    if px1 is not None and not px1.empty:
        # Normalize columns
        if 'Date' in px1.columns: px1 = px1.rename(columns={'Date': 'date'})
        if 'Close' in px1.columns: px1 = px1.rename(columns={'Close': 'close'})
        if 'Volume' in px1.columns: px1 = px1.rename(columns={'Volume': 'volume'})
        
        print(f"Shape: {px1.shape}")
        print(f"Columns: {list(px1.columns)}")
        print("Latest prices:")
        print(px1[['date', 'close', 'volume']].tail(3))
        latest_price1 = px1['close'].iloc[-1]
        print(f"Latest close: {latest_price1}")
    else:
        print("No data")
        latest_price1 = None
        
except Exception as e:
    print(f"Error: {e}")
    latest_price1 = None

print("=" * 50)
print("2. VnAdapter (from pick_best_by_symbols):")
try:
    from pick_best_by_symbols import VnAdapter
    adapter = VnAdapter()
    px2 = adapter.get_quote_history(symbol, days=30)
    
    print(f"Shape: {px2.shape}")
    print(f"Columns: {list(px2.columns)}")
    print("Latest prices:")
    print(px2[['date', 'close', 'volume']].tail(3))
    latest_price2 = px2['close'].iloc[-1]
    print(f"Latest close: {latest_price2}")
    
except Exception as e:
    print(f"Error: {e}")
    latest_price2 = None

print("=" * 50)
print("3. Comparison:")
if latest_price1 and latest_price2:
    ratio = latest_price2 / latest_price1
    print(f"Direct API price: {latest_price1}")
    print(f"VnAdapter price: {latest_price2}")
    print(f"Ratio: {ratio:.2f}")
    if abs(ratio - 10) < 1:
        print("→ VnAdapter trả về giá đã nhân 10!")
    elif abs(ratio - 1000) < 100:
        print("→ VnAdapter trả về giá đã nhân 1000!")
    else:
        print(f"→ Scale factor: {ratio:.1f}")
else:
    print("Could not compare - missing data")