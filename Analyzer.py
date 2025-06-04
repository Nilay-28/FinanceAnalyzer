import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import datetime
import warnings
from typing import List, Tuple
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# Initialize NLTK
@st.cache_resource
def get_sentiment_analyzer():
    try:
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except:
        return None

# UI Setup with enhanced styling
st.set_page_config(page_title="Market Sentiment Analyzer Pro", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠📈 Market Sentiment Analyzer Pro")
st.markdown("**Advanced ML-powered stock analysis with sentiment integration**")

# Enhanced Configuration
TICKERS = {
    "Apple (AAPL)": "AAPL", "Google (GOOGL)": "GOOGL", "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN", "Tesla (TSLA)": "TSLA", "Meta (META)": "META",
    "Nvidia (NVDA)": "NVDA", "Netflix (NFLX)": "NFLX", "Adobe (ADBE)": "ADBE",
    "Salesforce (CRM)": "CRM"
}

# Enhanced Sidebar
with st.sidebar:
    st.header("📊 Configuration")
    
    selected = st.selectbox("Choose Stock", list(TICKERS.keys()))
    ticker = TICKERS[selected]
    company_name = selected.split("(")[0].strip()
    
    custom = st.text_input("Or custom ticker:", placeholder="e.g., GOOG")
    if custom:
        ticker = custom.upper()
        company_name = custom.upper()
    
    investment = st.number_input("Investment (USD)", 100.0, value=1000.0, step=100.0)
    model_type = st.selectbox("Model", ["Random Forest", "Linear Regression"])
    
    test_size = 0.2
    lookback = 5
    news_count = 20

    
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    default_start = today - datetime.timedelta(days=365)
    
    start_date = st.date_input("Start", default_start, max_value=yesterday)
    end_date = st.date_input("End", yesterday, max_value=yesterday)
    
    if start_date >= end_date:
        st.error("Start date must be before end date")
        st.stop()

# Enhanced data loading with multiple fallback methods
@st.cache_data(ttl=300)
def load_data(symbol, start, end):
    def standardize_columns(df):
        if df.empty:
            return df
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        column_mapping = {'Adj Close': 'Adj_Close'}
        df = df.rename(columns=column_mapping)
        
        if 'Adj_Close' not in df.columns and 'Close' in df.columns:
            df['Adj_Close'] = df['Close']
        
        df.reset_index(inplace=True)
        return df.dropna()
    
    # Method 1: Standard download
    try:
        df = yf.download(symbol, start=start, end=end, progress=False, timeout=10)
        if not df.empty:
            df = standardize_columns(df)
            if not df.empty and 'Adj_Close' in df.columns:
                return df
    except Exception as e:
        st.warning(f"⚠️ Primary data download failed for {symbol}: {e}")

    # Method 2: Using Ticker object
    try:
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(start=start, end=end, timeout=10)
        if not df.empty:
            df = standardize_columns(df)
            if not df.empty and 'Adj_Close' in df.columns:
                return df
    except Exception as e:
        st.warning(f"⚠️ Fallback method using Ticker object failed for {symbol}: {e}")

    # If both fail and df is still empty
    st.error(f"❌ '{symbol}' is not a valid ticker or has no available data between {start} and {end}.")
    st.info(f"🔁 Using sample data instead.")
    return generate_sample_data(symbol, start, end)

def generate_sample_data(symbol, start, end):
    dates = pd.date_range(start, end, freq='D')
    dates = dates[dates.weekday < 5]
    
    if len(dates) == 0:
        return pd.DataFrame()
    
    np.random.seed(hash(symbol) % 2147483647)
    base_prices = {'AAPL': 180, 'GOOGL': 140, 'MSFT': 380, 'AMZN': 145, 'TSLA': 250}
    base_price = base_prices.get(symbol, 100)
    
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'Date': dates, 'Open': prices, 'Close': prices, 'Adj_Close': prices,
        'High': [p * 1.02 for p in prices], 'Low': [p * 0.98 for p in prices],
        'Volume': np.random.randint(1000000, 50000000, len(dates))
    })
    
    st.info(f"📊 Using sample data for {symbol} - This is for demonstration only!")
    return df

# Enhanced news fetching
@st.cache_data(ttl=900)
def fetch_news(company, count=20):
    # Try NewsAPI if available
    try:
        if hasattr(st, 'secrets') and "news_api_key" in st.secrets:
            api_key = st.secrets["news_api_key"]
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": f'"{company}" stock OR shares',
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": count,
                "apiKey": api_key
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                articles = response.json().get("articles", [])
                titles = [article["title"] for article in articles if article.get("title")]
                if titles:
                    return titles
    except:
        pass
    
    # Fallback sample news
    sample_news = [
        f"{company} reports quarterly earnings beat expectations",
        f"{company} stock shows strong momentum in tech sector",
        f"Analysts upgrade {company} rating following performance",
        f"{company} announces new product developments",
        f"Market outlook remains positive for {company}",
        f"{company} demonstrates resilience in volatile market"
    ]
    st.info("📰 Using sample news headlines for demonstration")
    return sample_news[:count//2]

# Enhanced technical indicators
def add_features(df, lookback=10):
    df = df.copy()
    
    if 'Adj_Close' not in df.columns:
        st.error("Missing Adj_Close column")
        return df
    
    # Basic features
    df["Return"] = df["Adj_Close"].pct_change()
    df["Log_Return"] = np.log(df["Adj_Close"] / df["Adj_Close"].shift(1))
    
    # Moving averages
    df["SMA_5"] = df["Adj_Close"].rolling(5).mean()
    df["SMA_20"] = df["Adj_Close"].rolling(min(20, len(df)//2)).mean()
    df["EMA_12"] = df["Adj_Close"].ewm(span=12).mean()
    
    # Advanced indicators
    df["Volatility"] = df["Return"].rolling(lookback).std()
    df["RSI"] = calculate_rsi(df["Adj_Close"])
    df["High_Low_Pct"] = (df["High"] - df["Low"]) / df["Low"] if 'High' in df.columns else 0
    df["Price_Change"] = df["Adj_Close"].pct_change(periods=lookback)
    
    # Volume features
    if 'Volume' in df.columns:
        df["Volume_SMA"] = df["Volume"].rolling(lookback).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]
    else:
        df["Volume_Ratio"] = 1
    
    return df

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Enhanced sentiment analysis
def analyze_sentiment(titles):
    sia = get_sentiment_analyzer()
    if not sia or not titles:
        return 0.0, [0.0], "No sentiment data"
    
    scores = []
    for title in titles:
        try:
            score = sia.polarity_scores(title)["compound"]
            scores.append(score)
        except:
            scores.append(0.0)
    
    avg = np.mean(scores) if scores else 0.0
    
    if avg > 0.3:
        label = "Positive 📈"
    elif avg < -0.3:
        label = "Negative 📉"
    else:
        label = "Neutral ➖"
    
    return avg, scores, label

# Load and process data
with st.spinner("Loading stock data..."):
    df = load_data(ticker, start_date, end_date)

if df.empty or len(df) < 30:
    st.error("❌ Insufficient data. Try different dates or ticker.")
    st.stop()

st.success(f"📊 Loaded {len(df)} trading days for {ticker}")
data_range = f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
st.info(f"**Data Range:** {data_range}")

# News and sentiment
with st.spinner("Analyzing sentiment..."):
    news = fetch_news(company_name, news_count)
    avg_sentiment, sentiments, sentiment_label = analyze_sentiment(news)

# Enhanced metrics display
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Current Price", f"${df['Adj_Close'].iloc[-1]:.2f}")
with col2:
    change = df['Adj_Close'].pct_change().iloc[-1] * 100
    st.metric("Daily Change", f"{change:.2f}%", delta=f"{change:.2f}%")
with col3:
    st.metric("Sentiment Score", f"{avg_sentiment:.3f}")
with col4:
    st.metric("Sentiment", sentiment_label)

# News display with sentiment scoring
st.subheader("📰 Recent News & Sentiment")
if news:
    news_df = pd.DataFrame({
        "Headline": news[:8],
        "Sentiment": sentiments[:8]
    })
    
    def color_sentiment(val):
        if val > 0.1:
            return 'background-color: lightgreen'
        elif val < -0.1:
            return 'background-color: lightcoral'
        return 'background-color: lightgray'
    
    styled_news = news_df.style.applymap(color_sentiment, subset=['Sentiment'])
    st.dataframe(styled_news, use_container_width=True)

# Feature engineering
df_features = add_features(df, lookback)
df_features["Sentiment"] = avg_sentiment
df_features["Lag1_Sentiment"] = df_features["Sentiment"].shift(1)
df_features.dropna(inplace=True)

if len(df_features) < 20:
    st.error("❌ Insufficient processed data")
    st.stop()

# Enhanced model training
feature_columns = ["Lag1_Sentiment", "Volatility", "SMA_5", "RSI", "High_Low_Pct", "Volume_Ratio", "Price_Change"]
available_features = [f for f in feature_columns if f in df_features.columns]

X = df_features[available_features]
y = df_features["Return"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
if model_type == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
else:
    model = LinearRegression()

model.fit(X_train_scaled, y_train)

# Enhanced predictions and metrics
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

st.subheader("🎯 Model Performance")
perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
with perf_col1:
    st.metric("Train R²", f"{train_r2:.4f}")
with perf_col2:
    st.metric("Test R²", f"{test_r2:.4f}")
with perf_col3:
    st.metric("MAE", f"{test_mae:.4f}")
with perf_col4:
    st.metric("RMSE", f"{test_rmse:.4f}")

if test_r2 < 0.05:
    st.warning("⚠️ Low model performance. Predictions may be unreliable.")

# Feature importance for Random Forest
if model_type == "Random Forest":
    st.subheader("🎯 Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': available_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig_imp, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
    ax.set_title('Feature Importance')
    st.pyplot(fig_imp)

# Enhanced portfolio simulation
X_full_scaled = scaler.transform(X)
df_features["Predicted_Return"] = model.predict(X_full_scaled)

# Portfolio with transaction costs
transaction_cost = 0.001
df_features["Strategy_Return"] = df_features["Predicted_Return"] - transaction_cost
df_features["Cumulative_Actual"] = (1 + df_features["Return"]).cumprod()
df_features["Cumulative_Strategy"] = (1 + df_features["Strategy_Return"]).cumprod()

portfolio_actual = investment * df_features["Cumulative_Actual"]
portfolio_strategy = investment * df_features["Cumulative_Strategy"]

st.subheader("💼 Portfolio Performance")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_features["Date"], portfolio_actual, label="Buy & Hold", linewidth=2)
ax.plot(df_features["Date"], portfolio_strategy, label="ML Strategy", linewidth=2)
ax.set_title("Portfolio Comparison")
ax.set_ylabel("Portfolio Value (USD)")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Portfolio returns
actual_return = (portfolio_actual.iloc[-1] - investment) / investment * 100
strategy_return = (portfolio_strategy.iloc[-1] - investment) / investment * 100

ret_col1, ret_col2, ret_col3 = st.columns(3)
with ret_col1:
    st.metric("Buy & Hold Return", f"{actual_return:.2f}%")
with ret_col2:
    st.metric("ML Strategy Return", f"{strategy_return:.2f}%")
with ret_col3:
    excess = strategy_return - actual_return
    st.metric("Excess Return", f"{excess:.2f}%", delta=f"{excess:.2f}%")

# Enhanced trading signals
latest_pred = df_features["Predicted_Return"].iloc[-1]
signal_strength = abs(latest_pred) + abs(avg_sentiment) / 2

if latest_pred > 0.01 and avg_sentiment > 0.1:
    signal = "🟢 STRONG BUY"
elif latest_pred > 0.005 or avg_sentiment > 0.05:
    signal = "🟡 BUY"
elif latest_pred < -0.01 and avg_sentiment < -0.1:
    signal = "🔴 STRONG SELL"
elif latest_pred < -0.005 or avg_sentiment < -0.05:
    signal = "🟠 SELL"
else:
    signal = "⚪ HOLD"

st.subheader("🚦 Trading Signals")
st.markdown(f"**Current Signal:** {signal}")
st.markdown(f"**Signal Strength:** {signal_strength:.3f}")

# Enhanced risk metrics
returns = df_features["Return"].dropna()
volatility = returns.std() * np.sqrt(252)
sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
max_drawdown = (portfolio_actual / portfolio_actual.cummax() - 1).min()

st.subheader("⚠️ Risk Analysis")
risk_col1, risk_col2, risk_col3 = st.columns(3)
with risk_col1:
    st.metric("Annualized Volatility", f"{volatility:.2%}")
with risk_col2:
    st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
with risk_col3:
    st.metric("Max Drawdown", f"{max_drawdown:.2%}")

# Optional detailed view with export
if st.checkbox("🔍 Show Detailed Analysis"):
    st.subheader("📊 Recent Data")
    display_cols = ["Date", "Adj_Close", "Return", "Predicted_Return", "Sentiment", "RSI"]
    available_cols = [col for col in display_cols if col in df_features.columns]
    st.dataframe(df_features[available_cols].tail(15), use_container_width=True)
    
    csv = df_features.to_csv(index=False)
    st.download_button(
        "📥 Download Full Dataset", 
        csv, 
        f"{ticker}_analysis_{datetime.date.today()}.csv", 
        "text/csv"
    )

st.markdown("---")
st.markdown("**Disclaimer:** Educational purposes only. Not financial advice. Always consult financial professionals before investing.")