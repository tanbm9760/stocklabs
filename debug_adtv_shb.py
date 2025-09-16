#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pick_best_by_symbols import VnAdapter, _calc_adtv_vnd

symbol = 'SHB'
print(f'Debugging ADTV calculation for {symbol}...')

try:
    adapter = VnAdapter()
    px = adapter.get_quote_history(symbol, days=30)
    
    # Show last 5 days raw data
    print("Raw price data (last 5 days):")
    print(px.tail(5)[['date', 'close', 'volume']])
    
    # Calculate ADTV step by step
    print(f"\nADTV calculation debug:")
    last_20_days = px.tail(20).copy()
    last_20_days["close"] = pd.to_numeric(last_20_days["close"], errors="coerce")
    last_20_days["volume"] = pd.to_numeric(last_20_days["volume"], errors="coerce")
    
    # BEFORE multiplier
    tv_before = (last_20_days["close"] * last_20_days["volume"]).dropna()
    adtv_before = float(tv_before.mean()) if len(tv_before) else np.nan
    
    # AFTER multiplier  
    tv_after = (last_20_days["close"] * 1000 * last_20_days["volume"]).dropna()
    adtv_after = float(tv_after.mean()) if len(tv_after) else np.nan
    
    print(f"ADTV WITHOUT *1000: {adtv_before:,.0f} VND = {adtv_before/1e9:.1f} tỷ")
    print(f"ADTV WITH *1000: {adtv_after:,.0f} VND = {adtv_after/1e9:.1f} tỷ")
    
    # Check individual day calculations
    print(f"\nSample calculations (last 3 days):")
    for i in range(3):
        idx = -(i+1)
        date = last_20_days['date'].iloc[idx]
        price = last_20_days['close'].iloc[idx] 
        volume = last_20_days['volume'].iloc[idx]
        tv_no_mult = price * volume
        tv_with_mult = price * 1000 * volume
        print(f"  {date}: {price:,.2f} × {volume:,.0f} = {tv_no_mult:,.0f} → {tv_with_mult:,.0f} VND")
    
    # Test official function
    official_adtv = _calc_adtv_vnd(px, n=20)
    print(f"\nOfficial _calc_adtv_vnd: {official_adtv:,.0f} VND = {official_adtv/1e9:.1f} tỷ")
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()