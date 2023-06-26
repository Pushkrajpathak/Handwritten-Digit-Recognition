import streamlit as st
import yfinance as yf
import pandas as pd

st.write("""
# Simple stock price APP

Shown are stock price and volume of google stock

""")
tickersymbol='GOOGL'
ticker_data=yf.Ticker(tickersymbol)

tickerDF=ticker_data.history(period='id',start='2022-5-31',end='2023-5-31')

st.line_chart(tickerDF.Close)
st.line_chart(tickerDF.Volume)