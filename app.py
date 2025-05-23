# app.py

import math
import time
import datetime
from io import BytesIO

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pyotp
from prophet import Prophet
from scipy.signal import argrelextrema
from SmartApi.smartConnect import SmartConnect


st.set_page_config(page_title="Wave & Prophet Scanner", layout="wide")

# --- Angel One credentials (hard‐coded) ---
API_KEY     = "EKa93pFu"
CLIENT_ID   = "R59803990"
PASSWORD    = "1234"
TOTP_SECRET = "5W4MC6MMLANC3UYOAW2QDUIFEU"

# --- 1) Load token map from CSV ---
@st.cache_data
def load_token_map(path="nse_stock_tokens.csv"):
    df = pd.read_csv(path)
    return {f"{r.symbol}.NS": str(r.token) for r in df.itertuples()}

token_map   = load_token_map()
all_symbols = list(token_map.keys())
total       = len(all_symbols)

# --- 2) Batch selector (500 symbols per batch) ---
batch_size   = 500
n_batches    = math.ceil(total / batch_size)
batch_labels = [
    f"{i} ({(i-1)*batch_size+1}–{min(i*batch_size, total)})"
    for i in range(1, n_batches+1)
]
batch_choice = st.sidebar.selectbox("Batch #", batch_labels, index=0)
batch_num    = int(batch_choice.split()[0])
symbols      = all_symbols[(batch_num-1)*batch_size : batch_num*batch_size]
st.sidebar.markdown(f"Total symbols: {total} — batch {batch_num}/{n_batches}")

# --- 3) Cached Angel One login ---
@st.cache_resource
def angel_login():
    totp   = pyotp.TOTP(TOTP_SECRET).now()
    client = SmartConnect(api_key=API_KEY)
    client.generateSession(CLIENT_ID, PASSWORD, totp)
    return client

client = angel_login()

# --- 4) Fetch historical OHLCV (throttle 0.3s) ---
@st.cache_data
def fetch_price_data(token: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    sd, ed = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    params = {
        "exchange":    "NSE",
        "symboltoken": token,
        "interval":    "ONE_DAY",
        "fromdate":    f"{sd} 00:00",
        "todate":      f"{ed} 23:59"
    }
    resp = client.getCandleData(params)
    time.sleep(0.3)
    df = pd.DataFrame(resp["data"], columns=['Date','Open','High','Low','Close','Volume'])
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_convert(None)
    return df

# --- 5) Elliott Wave detection ---
def identify_elliott_wave(df: pd.DataFrame):
    close = df['Close'].values
    idxs = np.sort(np.concatenate([
        argrelextrema(close, np.greater, order=5)[0],
        argrelextrema(close, np.less,   order=5)[0]
    ]))
    if len(idxs) < 3:
        return None
    w1, w2, w3 = idxs[-3:]
    if close[w2] <= close[w1]:
        return None
    wave1 = close[w2] - close[w1]
    wave2 = close[w2] - close[w3]
    retrace = wave2 / wave1
    if not (0.382 <= retrace <= 0.618):
        return None
    prob = (1 - abs(retrace - 0.5)/0.118) * 100
    prob = max(50, min(prob, 90))
    vol1 = df['Volume'].iloc[w1:w2].mean()
    vol3 = df['Volume'].iloc[w3:].mean()
    return {
        'probability':        prob,
        'retracement':        retrace,
        'wave1_start_date':   df['Date'].iloc[w1].date(),
        'wave1_end_date':     df['Date'].iloc[w2].date(),
        'wave2_end_date':     df['Date'].iloc[w3].date(),
        'last_extrema_idx':   np.array([w1, w2, w3]),
        'last_extrema_prices':close[[w1, w2, w3]],
        'volume_confirmation':vol3 > vol1
    }

# --- 6) Plotly Elliott Wave chart ---
def get_wave_chart(df, wave, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'], mode='lines', name='Close',
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>'
    ))
    dates  = df['Date'].iloc[wave['last_extrema_idx']]
    prices = wave['last_extrema_prices']
    fig.add_trace(go.Scatter(
        x=dates, y=prices, mode='markers+lines', name='Wave Points',
        marker=dict(size=8, color='red'),
        hovertemplate='Wave Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>'
    ))
    vm = 'with' if wave['volume_confirmation'] else 'without'
    fig.update_layout(
        title=f"{ticker} Elliott Wave · Prob {wave['probability']:.1f}% {vm} vol",
        xaxis_title='Date', yaxis_title='Price',
        hovermode='x unified', height=350
    )
    return fig

# --- 7) Plotly Prophet chart (wave2_date optional) ---
def get_prophet_chart(df, ticker, wave2_date=None):
    dfp = df[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
    m   = Prophet(daily_seasonality=True)
    m.fit(dfp)

    future = m.make_future_dataframe(periods=0)
    fc     = m.predict(future)

    fig = go.Figure()
    # Actual points
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'], mode='markers',
        marker=dict(color='black', size=6),
        name='Actual',
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>'
    ))
    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc['ds'], y=fc['yhat'], mode='lines',
        line=dict(color='blue'), name='Forecast',
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Fit: %{y:.2f}<extra></extra>'
    ))
    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([fc['ds'], fc['ds'][::-1]]),
        y=pd.concat([fc['yhat_upper'], fc['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(173,216,230,0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Interval',
        hoverinfo='skip'
    ))
    # Optional Wave 2 marker
    if wave2_date is not None:
        price_w2 = df.loc[df['Date'].dt.date == wave2_date, 'Close'].iloc[0]
        fig.add_trace(go.Scatter(
            x=[wave2_date], y=[price_w2], mode='markers',
            marker=dict(symbol='star', size=14, color='orange', line=dict(width=1, color='black')),
            name='Wave 2 End',
            hovertemplate='Wave 2 End: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=f"{ticker} Prophet Fit{'' if wave2_date is None else ' · Wave 2 Marker'}",
        xaxis_title='Date', yaxis_title='Price',
        hovermode='x unified', height=350
    )
    return fig, fc

# --- 8) Plotly Outlier chart ---
def get_outlier_chart(dfc, fc2, dsel, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dfc['Date'], y=dfc['Close'], mode='lines+markers',
        line=dict(color='black'),
        marker=dict(size=5, color='black'),
        name='Actual',
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=fc2['ds'], y=fc2['yhat'], mode='lines',
        line=dict(color='blue'),
        name='Forecast',
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Pred: %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([fc2['ds'], fc2['ds'][::-1]]),
        y=pd.concat([fc2['yhat_upper'], fc2['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(173,216,230,0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True, name='95% Interval', hoverinfo='skip'
    ))
    out_val = dfc[dfc['Date'].dt.date==dsel]['Close'].iloc[0]
    fig.add_trace(go.Scatter(
        x=[dsel], y=[out_val], mode='markers',
        marker=dict(color='red', size=10, symbol='x'),
        name='Outlier',
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Outlier: %{y:.2f}<extra></extra>'
    ))
    fig.update_layout(
        title=f"{ticker} Outlier on {dsel}",
        xaxis_title='Date', yaxis_title='Price',
        hovermode='x unified', height=350
    )
    return fig

# --- 9) CSV helper ---
def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 10) Three tabs ---
tab1, tab2, tab3 = st.tabs([
    "🏠 Home (Wave Scan)",
    "📈 Prophet Only",
    "⚠️ Outlier Filter"
])

# Tab 1: Home
with tab1:
    st.header("Elliott Wave Scanner")
    start_date = st.date_input("Start date", datetime.date.today() - datetime.timedelta(days=90), key="w1")
    end_date = st.date_input("End date", datetime.date.today(), key="w2")

    if st.button("▶️ Run wave scan"):
        wave_results = []
        prog = st.progress(0)
        for i, sym in enumerate(symbols, start=1):
            df = fetch_price_data(token_map[sym], start_date, end_date)
            w = identify_elliott_wave(df)
            if w:
                wave_results.append({'Ticker': sym, **w})
            prog.progress(i / len(symbols))
            time.sleep(0.4)  # 👈 Throttle to avoid API rate limit

        if not wave_results:
            st.info("No stocks met the Elliott wave criteria.")
        else:
            df_wave = pd.DataFrame(wave_results)
            df_wave = df_wave[[
                'Ticker', 'probability', 'retracement',
                'wave1_start_date', 'wave1_end_date', 'wave2_end_date',
                'volume_confirmation'
            ]]
            df_wave.columns = [
                'Ticker', 'Probability', 'Retracement',
                'Wave1Start', 'Wave1End', 'Wave2End',
                'VolConfirm'
            ]
            df_wave.sort_values('Probability', ascending=False, inplace=True)

            st.subheader("Wave Scan Results")
            st.dataframe(df_wave, use_container_width=True)
            st.download_button(
                "📥 Download wave results CSV",
                to_csv_bytes(df_wave),
                "wave_results.csv", "text/csv"
            )

            for row in df_wave.itertuples(index=False):
                sym = row.Ticker
                st.markdown(f"### {sym} · Prob {row.Probability:.1f}%")
                dfc = fetch_price_data(token_map[sym], start_date, end_date)
                wave = identify_elliott_wave(dfc)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Elliott Wave**")
                    fig1 = get_wave_chart(dfc, wave, sym)
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    st.markdown("**Prophet Fit (Wave 2 End)**")
                    fig2, _ = get_prophet_chart(
                        dfc, sym, wave['wave2_end_date']
                    )
                    st.plotly_chart(fig2, use_container_width=True)

# Tab 2: Prophet Only
with tab2:
    st.header("Prophet Only")
    p_start = st.date_input("Start date", datetime.date.today() - datetime.timedelta(days=90), key="p1")
    p_end = st.date_input("End date", datetime.date.today(), key="p2")
    ticker = st.selectbox("Ticker", ["-- Select --"] + symbols, key="p3")

    if ticker != "-- Select --" and st.button("▶️ Generate forecast", key="p4"):
        dfp = fetch_price_data(token_map[ticker], p_start, p_end)
        time.sleep(0.4)  # 👈 Throttle API call
        if not dfp.empty:
            wave_data = identify_elliott_wave(dfp)
            wave2_date = wave_data['wave2_end_date'] if wave_data else None
            fig, fc2 = get_prophet_chart(dfp, ticker, wave2_date)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Forecast Data")
            st.dataframe(fc2, use_container_width=True)
            st.download_button(
                "📥 Download forecast CSV",
                to_csv_bytes(fc2),
                f"{ticker}_forecast.csv", "text/csv"
            )
        else:
            st.warning("No data returned.")

# Tab 3: Outlier Filter
with tab3:
    st.header("Outlier Filter")
    o_start = st.date_input("History start", datetime.date.today() - datetime.timedelta(days=90), key="o1")
    o_end = st.date_input("History end", datetime.date.today(), key="o2")

    if st.button("▶️ Find outliers", key="o3"):
        outliers = []
        prog = st.progress(0)
        for i, sym in enumerate(symbols, start=1):
            df = fetch_price_data(token_map[sym], o_start, o_end)
            time.sleep(0.4)  # 👈 Add delay between API requests
            df['ds'] = df['Date'].dt.normalize()
            dfp = df[['ds', 'Close']].rename(columns={'ds': 'ds', 'Close': 'y'})
            m = Prophet(daily_seasonality=True)
            m.fit(dfp)
            fc2 = m.predict(m.make_future_dataframe(periods=0))
            merged = pd.merge(
                df[['ds', 'Close']],
                fc2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                on='ds', how='inner'
            )
            bad = merged[
                (merged.Close < merged.yhat_lower) |
                (merged.Close > merged.yhat_upper)
            ]
            for _, r in bad.iterrows():
                outliers.append({
                    'Ticker': sym,
                    'Date': r.ds.date(),
                    'Actual': r.Close,
                    'Lower': r.yhat_lower,
                    'Upper': r.yhat_upper,
                    'Forecast': r.yhat
                })
            prog.progress(i / len(symbols))

        if not outliers:
            st.info("No outliers detected in that batch.")
        else:
            df_out = pd.DataFrame(outliers)
            df_latest = df_out.sort_values('Date').groupby('Ticker', as_index=False).last()

            st.subheader("Latest Outlier per Ticker")
            st.dataframe(df_latest[['Ticker', 'Date', 'Actual', 'Lower', 'Upper', 'Forecast']], use_container_width=True)
            st.download_button(
                "📥 Download latest outliers CSV",
                to_csv_bytes(df_latest),
                "latest_outliers.csv", "text/csv"
            )

            for row in df_latest.itertuples(index=False):
                ticker = row.Ticker
                dsel = row.Date
                st.markdown(f"### {ticker} · Outlier on {dsel}")
                dfc = fetch_price_data(token_map[ticker], o_start, o_end)
                time.sleep(0.4)  # 👈 Add delay again
                dfc['ds'] = dfc['Date'].dt.normalize()
                dfp2 = dfc[['ds', 'Close']].rename(columns={'ds': 'ds', 'Close': 'y'})
                m2 = Prophet(daily_seasonality=True)
                m2.fit(dfp2)
                fc2 = m2.predict(m2.make_future_dataframe(periods=0))

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Prophet Forecast**")
                    fig_pf, _ = get_prophet_chart(dfc, ticker)
                    st.plotly_chart(fig_pf, use_container_width=True)
                with col2:
                    st.markdown("**Outlier Detail**")
                    fig_ot = get_outlier_chart(dfc, fc2, dsel, ticker)
                    st.plotly_chart(fig_ot, use_container_width=True)
