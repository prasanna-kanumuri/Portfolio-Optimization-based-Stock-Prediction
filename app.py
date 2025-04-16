import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date


# Load the model
model = load_model('Stock_Predictions_Model.h5')

# Custom Styles for Modern Background
# Custom Styles for Modern Background
def apply_styles():
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(to bottom, #ffffff, #f0f4f8);
            font-family: 'Arial', sans-serif;
            color: #333333;
        }
        .main-title {
            font-size: 36px;
            color: #1d3557;
            text-align: center;
            margin-bottom: 20px;
            font-weight: bold;
        }
        .section-title {
            font-size: 26px;
            color: #457b9d;
            margin-top: 20px;
            font-weight: bold;
        }
        .stock-table {
            margin: auto;
            border: 1px solid #ddd;
        }
        .success-box {
            background-color: #28a745; /* Stronger green */
            color: white; /* Clearer text color */
            padding: 15px; /* Increased padding for clarity */
            border-radius: 10px; /* Rounded corners */
            border-left: 5px solid #155724; /* Darker green for emphasis */
            font-weight: bold; /* Make text bold */
            font-size: 18px; /* Slightly larger text */
            margin-bottom: 10px;
        }
        .info-box {
            background-color: #d1ecf1;
            padding: 10px;
            border-radius: 5px;
            border-left: 5px solid #17a2b8;
            margin-bottom: 10px;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            border-left: 5px solid #ffc107;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_styles()

# Streamlit app header
st.markdown('<div class="main-title">Enhanced Stock Market Predictor</div>', unsafe_allow_html=True)

# Section: Predict Tomorrow's Stock Price
st.markdown('<div class="section-title">Predict Tomorrow\'s Stock Price</div>', unsafe_allow_html=True)
stock = st.text_input('Enter Stock Symbol for Prediction', '^NSEI')

start = '2012-01-01'
end = date.today().strftime('%Y-%m-%d')
data = yf.download(stock, start=start, end=end)

if not data.empty:
    st.subheader('Stock Data (Until Today)')
    st.write(data)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Prepare input for tomorrow's prediction
    last_100_days = data_scaled[-100:]
    X_test = np.array([last_100_days])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predict tomorrow's price
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    st.markdown(
        f'<div class="success-box">Tomorrow\'s Predicted Price for <b>{stock}</b>: {predicted_price[0][0]:.2f}</div>',
        unsafe_allow_html=True,
    )

    # Visualization Section
    st.markdown('<div class="section-title">Visualizations</div>', unsafe_allow_html=True)

    # Simplified Moving Average Graph (MA100)
    st.subheader('Price vs MA100')
    ma_100_days = data['Close'].rolling(100).mean()
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label='Close Price', color='green')
    plt.plot(ma_100_days, label='MA100', color='blue')
    plt.legend()
    plt.title('Close Price vs MA100')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig1)

    # Daily Percentage Change Graph
    st.subheader('Daily Percentage Change')
    daily_pct_change = data['Close'].pct_change() * 100
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(daily_pct_change, label='Daily % Change', color='purple')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.title('Daily Percentage Change in Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Percentage Change (%)')
    st.pyplot(fig2)

    # Cumulative Returns Graph
    st.subheader('Cumulative Returns')
    cumulative_returns = (1 + daily_pct_change / 100).cumprod()
    fig3 = plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label='Cumulative Returns', color='orange')
    plt.legend()
    plt.title('Cumulative Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    st.pyplot(fig3)

else:
    st.markdown('<div class="warning-box">No data found. Please check the stock symbol and try again.</div>', unsafe_allow_html=True)

# Section: Compare Multiple Stocks
st.markdown('<div class="section-title">Compare Stocks</div>', unsafe_allow_html=True)
stocks_to_compare = st.text_input('Enter up to 3 Stock Symbols (comma-separated)', 'AAPL, MSFT, TSLA')

comparison_data = []  # Initialize the comparison data list

if stocks_to_compare:
    stock_list = [s.strip() for s in stocks_to_compare.split(',')[:3]]  # Limit to 3 stocks
    for stock in stock_list:
        data = yf.download(stock, start='2020-01-01', end=end)
        if not data.empty:
            comparison_data.append({
                'Stock': stock,
                'Latest Price': data['Close'].iloc[-1],
                'Price 6 Months Ago': data['Close'].iloc[0],
                '6-Month Growth (%)': ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
            })

    # Check if we have any valid data
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)

        # Format numeric columns
        comparison_df['Latest Price'] = comparison_df['Latest Price'].astype(float).round(2)
        comparison_df['Price 6 Months Ago'] = comparison_df['Price 6 Months Ago'].astype(float).round(2)
        comparison_df['6-Month Growth (%)'] = comparison_df['6-Month Growth (%)'].astype(float).round(2)

        # Highlight the best performer
        st.subheader('Stock Comparison Table')
        st.dataframe(comparison_df.style.highlight_max(subset=['6-Month Growth (%)'], color='green', axis=0))

import pandas as pd
from datetime import datetime, timedelta

# Define the list of top Indian stock tickers (NSE format)
# Define the date range
end_date = datetime.today()
start_date = end_date - timedelta(days=180)  # Last 6 months

# Fetch all Indian stocks available in yfinance (this might require a precompiled list of tickers)
def fetch_all_indian_stocks():
    # This is an example list. A comprehensive list can be obtained from NSE/BSE or a dataset.
    all_indian_stocks = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS',
        'ITC.NS', 'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS',
        'ADANIGREEN.NS', 'DMART.NS', 'WIPRO.NS', 'TECHM.NS', 'LT.NS',
        'ULTRACEMCO.NS', 'MARUTI.NS', 'BAJAJFINSV.NS', 'AXISBANK.NS', 'KOTAKBANK.NS'
    ]
    return all_indian_stocks

# Define the list of all Indian stock tickers
all_tickers = fetch_all_indian_stocks()

# Fetch historical data for the past 6 months
data = yf.download(all_tickers, start=start_date, end=end_date)

# Check available columns and use 'Close' instead of 'Adj Close' if necessary
if 'Adj Close' in data:
    adj_close_data = data['Adj Close']
elif 'Close' in data:
    adj_close_data = data['Close']
else:
    raise KeyError("Neither 'Adj Close' nor 'Close' column found in downloaded data")

# Calculate the percentage change over the last 6 months
returns = (adj_close_data.iloc[-1] / adj_close_data.iloc[0] - 1) * 100

# Get the top 10 best-performing stocks based on percentage returns
top_performers = returns.sort_values(ascending=False).head(10)

top_performers_df = pd.DataFrame(top_performers, columns=['6M Return (%)'])

# Streamlit app header
st.title("Top 10 Recommended Indian Stocks for Investment")

# Introduction section
st.markdown("""
If you're looking to invest in the Indian stock market but aren't sure which stocks to pick, here are the top 10 best-performing stocks over the last 6 months based on their returns. These stocks have shown strong growth and could be good candidates for further research and investment.
""")

# Display the top 10 best-performing stocks
st.dataframe(top_performers_df)

# Save the data to CSV
top_performers_df.to_csv('top_10_indian_stocks_last_6_months.csv')

# Additional recommendation section
st.markdown("""
### Investment Tips:
1. **Diversify Your Portfolio**: Avoid putting all your money into a single stock.
2. **Do Your Research**: Look into the company's fundamentals and future growth prospects.
3. **Consult a Financial Advisor**: Always consult with a certified financial advisor before making investment decisions.
4. **Consider Risk Tolerance**: Ensure the stocks align with your risk appetite and investment goals.
""")
