import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px

# Load dataset
@st.cache_data
def load_data(koin):
    if koin == "BTC":
        df = pd.read_csv("Datasets/btc-usd-max.csv")
    elif koin == "ETH":
        df = pd.read_csv("Datasets/eth-usd-max.csv")

    df['snapped_at'] = pd.to_datetime(df['snapped_at'])
    df['koin'] = koin
    return df

# Title
st.title("₿ Dashboard BTC dan ETH Ξ")

# Sidebar
st.sidebar.header("Opsi Visualisasi")
opsi = st.sidebar.selectbox("Pilih Opsi", ["BTC", "ETH", "Perbandingan", "Prediksi"])

def filter_date(df, key_prefix=""):
    min_date = df['snapped_at'].min().date()
    max_date = df['snapped_at'].max().date()
    # d-m-Y format
    min_date = pd.to_datetime(min_date, format="%d-%m-%Y")
    max_date = pd.to_datetime(max_date, format="%d-%m-%Y")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Tanggal Mulai", value=min_date, min_value=min_date, max_value=max_date,
            key=f"{key_prefix}_start"
        )
    with col2:
        end_date = st.date_input(
            "Tanggal Akhir", value=max_date, min_value=min_date, max_value=max_date,
            key=f"{key_prefix}_end"
        )
    
    filtered = df[(df['snapped_at'].dt.date >= start_date) & (df['snapped_at'].dt.date <= end_date)]

    return filtered

# Visualisasi tunggal
if opsi == "BTC" or opsi == "ETH":
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Tabel Data", "Harga", "Kapitalisasi Pasar", "Volume Perdagangan", "Korelasi Antar-Fitur", "Distribusi Normal"
    ])
    
    df = load_data(opsi)
    with tab1:
        st.subheader(f"Tabel Data {opsi}")
        filtered_df = filter_date(df, "sample")
        st.dataframe(filtered_df[['snapped_at', 'price', 'market_cap', 'total_volume']])

    with tab2:
        st.subheader(f"📈 Harga {opsi} (USD)")
        filtered_df = filter_date(df, "harga")
        mean_price = filtered_df['price'].mean()
        fig = px.line(filtered_df, x='snapped_at', y='price', labels={'snapped_at': 'Tanggal', 'price': 'Harga (USD)'})
        fig.add_hline(y=mean_price, line_dash="dash", line_color="red", annotation_text="Rata-rata", annotation_position="top left")
        fig.update_traces(hovertemplate='%{x|%d %b %Y}<br>Harga: %{y:$,.2f}')
        fig.update_layout(xaxis_tickformat='%d %b %Y')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader(f"📈 Kapitalisasi Pasar {opsi} (USD)")
        filtered_df = filter_date(df, "cap")
        mean_cap = filtered_df['market_cap'].mean()
        fig = px.line(filtered_df, x='snapped_at', y='market_cap', labels={'snapped_at': 'Tanggal', 'market_cap': 'Market Cap (USD)'})
        fig.add_hline(y=mean_cap, line_dash="dash", line_color="red", annotation_text="Rata-rata", annotation_position="top left")
        fig.update_traces(hovertemplate='%{x|%d %b %Y}<br>Harga: %{y:$,.2f}')
        fig.update_layout(xaxis_tickformat='%d %b %Y')
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader(f"📈 Volume Perdagangan {opsi} (USD)")
        filtered_df = filter_date(df, "volume")
        mean_volume = filtered_df['total_volume'].mean()
        fig = px.line(filtered_df, x='snapped_at', y='total_volume', labels={'snapped_at': 'Tanggal', 'total_volume': 'Volume'})
        fig.add_hline(y=mean_volume, line_dash="dash", line_color="red", annotation_text="Rata-rata", annotation_position="top left")
        fig.update_traces(hovertemplate='%{x|%d %b %Y}<br>Harga: %{y:$,.2f}')
        fig.update_layout(xaxis_tickformat='%d %b %Y')
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader(f"Korelasi Antar-Fitur {opsi}")
        corr = df[['price', 'market_cap', 'total_volume']].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu', labels=dict(x="Fitur", y="Fitur", color="Korelasi"))
        fig.update_layout(title=f"Korelasi Antar-Fitur {opsi}", xaxis_title="Fitur", yaxis_title="Fitur")
        st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.subheader(f"📊 Distribusi Normal Harga {opsi}")
        filtered_df = filter_date(df, "dist")
        prices = filtered_df['price'].dropna()
        hist_fig = px.histogram(prices, nbins=30, opacity=0.6, histnorm='probability density', labels={'value': 'Harga (USD)'})
        hist_fig.update_layout(title=f"Distribusi Harga & Kurva Normal {opsi}", xaxis_title="Harga (USD)", yaxis_title="Density")
        st.plotly_chart(hist_fig, use_container_width=True)

    st.markdown(f"Sumber: [CoinGecko](https://www.coingecko.com/en/coins/{'bitcoin' if opsi == 'BTC' else 'ethereum'})")

# Prediksi harga
elif opsi == "Prediksi":
    st.subheader("📈 Prediksi Harga BTC dan ETH")
    st.write("Segera hadir!")

# Visualisasi perbandingan
else:
    df_all = pd.concat([load_data("BTC"), load_data("ETH")], ignore_index=True)

    tab1, tab2, tab3 = st.tabs([
        "Perbandingan Harga", "Perbandingan Market Cap", "Perbandingan Volume"
    ])

    with tab1:
        st.subheader("📈 Perbandingan Harga BTC vs ETH")
        filtered_df = filter_date(df_all, "harga")
        fig = px.line(filtered_df, x='snapped_at', y='price', color='koin', labels={'snapped_at': 'Tanggal', 'price': 'Harga (USD)', 'koin': 'Koin'})
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📈 Perbandingan Kapitalisasi Pasar")
        filtered_df = filter_date(df_all, "cap")
        fig = px.line(filtered_df, x='snapped_at', y='market_cap', color='koin', labels={'snapped_at': 'Tanggal', 'market_cap': 'Market Cap (USD)', 'koin': 'Koin'})
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("📈 Perbandingan Volume Perdagangan")
        filtered_df = filter_date(df_all, "volume")
        fig = px.line(filtered_df, x='snapped_at', y='total_volume', color='koin', labels={'snapped_at': 'Tanggal', 'total_volume': 'Volume', 'koin': 'Koin'})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"Sumber: [CoinGecko](https://www.coingecko.com)")