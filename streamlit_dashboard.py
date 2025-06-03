import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Load dataset
@st.cache_data
def load_data(koin):
    if koin == "BTC":
        df = pd.read_csv("Datasets/btc-github.csv")
        df = df.drop(df.columns[0], axis=1)
    elif koin == "ETH":
        df = pd.read_csv("Datasets/eth-github.csv")
        df = df.drop(df.columns[0], axis=1)

    df['snapped_at'] = pd.to_datetime(df['snapped_at'])
    df['koin'] = koin
    return df

def preprocess(df):
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.sort_values('snapped_at')
    df['market_cap'] = df['market_cap'].ffill()

    issue_cols = [col for col in ['issues_opened', 'issues_closed', 'issue_comments'] if col in df.columns]
    if issue_cols:
        df['issue_activity'] = df[issue_cols].sum(axis=1)
        df.drop(columns=issue_cols, inplace=True)

    pull_cols = [col for col in ['pulls_opened', 'pulls_merged', 'pulls_closed'] if col in df.columns]
    if pull_cols:
        df['pull_activity'] = df[pull_cols].sum(axis=1)
        df.drop(columns=pull_cols, inplace=True)

    df.drop(columns=[col for col in ['stars', 'forks'] if col in df.columns], inplace=True)
    return df

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

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
        "Data Koin", "Harga", "Kapitalisasi Pasar", "Volume Perdagangan", "Aktivitas GitHub", "Korelasi Antar-Fitur"
    ])
    
    df_raw = load_data(opsi)
    df = preprocess(df_raw.copy())
    with tab1:
        st.subheader(f"Data Koin {opsi}")
        filtered_df = filter_date(df, "sample")
        st.dataframe(filtered_df[['snapped_at', 'price', 'market_cap', 'total_volume']])
        st.markdown(f"Sumber: [CoinGecko](https://www.coingecko.com/en/coins/{'bitcoin' if opsi == 'BTC' else 'ethereum'})")

    with tab2:
        st.subheader(f"📈 Harga {opsi} (USD)")
        filtered_df = filter_date(df, "harga")
        mean_price = filtered_df['price'].mean()
        fig = px.line(filtered_df, x='snapped_at', y='price', labels={'snapped_at': 'Tanggal', 'price': 'Harga (USD)'})
        fig.add_hline(y=mean_price, line_dash="dash", line_color="red", annotation_text="Rata-rata", annotation_position="top left")
        fig.update_traces(hovertemplate='%{x|%d %b %Y}<br>Harga: %{y:$,.2f}')
        fig.update_layout(xaxis_tickformat='%d %b %Y')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"Sumber: [CoinGecko](https://www.coingecko.com/en/coins/{'bitcoin' if opsi == 'BTC' else 'ethereum'})")

    with tab3:
        st.subheader(f"📈 Kapitalisasi Pasar {opsi} (USD)")
        filtered_df = filter_date(df, "cap")
        mean_cap = filtered_df['market_cap'].mean()
        fig = px.line(filtered_df, x='snapped_at', y='market_cap', labels={'snapped_at': 'Tanggal', 'market_cap': 'Kapitalisasi Pasar (USD)'})
        fig.add_hline(y=mean_cap, line_dash="dash", line_color="red", annotation_text="Rata-rata", annotation_position="top left")
        fig.update_traces(hovertemplate='%{x|%d %b %Y}<br>Kapitalisasi Pasar: %{y:$,.2f}')
        fig.update_layout(xaxis_tickformat='%d %b %Y')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"Sumber: [CoinGecko](https://www.coingecko.com/en/coins/{'bitcoin' if opsi == 'BTC' else 'ethereum'})")

    with tab4:
        st.subheader(f"📈 Volume Perdagangan {opsi} (USD)")
        filtered_df = filter_date(df, "volume")
        mean_volume = filtered_df['total_volume'].mean()
        fig = px.line(filtered_df, x='snapped_at', y='total_volume', labels={'snapped_at': 'Tanggal', 'total_volume': 'Volume Perdagangan (USD)'})
        fig.add_hline(y=mean_volume, line_dash="dash", line_color="red", annotation_text="Rata-rata", annotation_position="top left")
        fig.update_traces(hovertemplate='%{x|%d %b %Y}<br>Volume Perdagangan: %{y:$,.2f}')
        fig.update_layout(xaxis_tickformat='%d %b %Y')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"Sumber: [CoinGecko](https://www.coingecko.com/en/coins/{'bitcoin' if opsi == 'BTC' else 'ethereum'})")
    
    with tab5:
        st.subheader(f"Aktivitas GitHub {opsi}")
        filtered_df = filter_date(df, "github")
        
        if 'issue_activity' in filtered_df.columns:
            fig = px.line(filtered_df, x='snapped_at', y='issue_activity', labels={'snapped_at': 'Tanggal', 'issue_activity': 'Aktivitas Isu'})
            mean_issues = filtered_df['issue_activity'].mean()
            fig.add_hline(y=mean_issues, line_dash="dash", line_color="red", annotation_text="Rata-rata", annotation_position="top left")
            fig.update_layout(title="Aktivitas Isu GitHub", xaxis_tickformat='%d %b %Y')
            st.plotly_chart(fig, use_container_width=True)

        if 'pull_activity' in filtered_df.columns:
            fig = px.line(filtered_df, x='snapped_at', y='pull_activity', labels={'snapped_at': 'Tanggal', 'pull_activity': 'Aktivitas Pull'})
            mean_pulls = filtered_df['pull_activity'].mean()
            fig.add_hline(y=mean_pulls, line_dash="dash", line_color="red", annotation_text="Rata-rata", annotation_position="top left")
            fig.update_layout(title="Aktivitas Pull Request GitHub", xaxis_tickformat='%d %b %Y')
            st.plotly_chart(fig, use_container_width=True)

        if 'commits' in filtered_df.columns:
            fig = px.line(filtered_df, x='snapped_at', y='commits', labels={'snapped_at': 'Tanggal', 'commits': 'Jumlah Commit'})
            mean_commits = filtered_df['commits'].mean()
            fig.add_hline(y=mean_commits, line_dash="dash", line_color="red", annotation_text="Rata-rata", annotation_position="top left")
            fig.update_layout(title="Jumlah Commit GitHub", xaxis_tickformat='%d %b %Y')
            st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.subheader(f"Korelasi Antar-Fitur {opsi}")

        corr1 = df_raw.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(corr1, text_auto=True, aspect="auto", color_continuous_scale='RdBu', labels=dict(x="Fitur", y="Fitur", color="Korelasi"))
        fig.update_layout(title=f"Korelasi Antar-Fitur {opsi} (Sebelum Preprocessing)", xaxis_title="Fitur", yaxis_title="Fitur")
        st.plotly_chart(fig, use_container_width=True)

        corr2 = df.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(corr2, text_auto=True, aspect="auto", color_continuous_scale='RdBu', labels=dict(x="Fitur", y="Fitur", color="Korelasi"))
        fig.update_layout(title=f"Korelasi Antar-Fitur {opsi} (Setelah Preprocessing)", xaxis_title="Fitur", yaxis_title="Fitur")
        st.plotly_chart(fig, use_container_width=True)

elif opsi == "Prediksi":
    st.subheader("📈 Prediksi Harga BTC dan ETH")
    koin_pred = st.selectbox("Pilih Koin", ["BTC", "ETH"])
    model_map = {
        "xgb": "XGBoost",
        "rf": "Random Forest",
        "dtree": "Decision Tree"
    }

    # Load harga aktual
    actual_file = f"Datasets/{koin_pred.lower()}-usd-max_21days.csv"
    df_actual = pd.read_csv(actual_file, index_col=0)
    df_actual.index = pd.to_datetime(df_actual.index, errors='coerce')
    df_actual = df_actual.rename_axis('snapped_at').reset_index()
    df_actual = df_actual.dropna(subset=['snapped_at'])
    df_actual['snapped_at'] = df_actual['snapped_at'].dt.tz_localize(None)

    df_pred = None
    eval_results = {}

    for short_model, full_model_name in model_map.items():
        # Load prediksi
        file_pred = f"Datasets/Prediction/{koin_pred.lower()}-{short_model}-21d-price.csv"
        temp_df = pd.read_csv(file_pred, index_col=0)
        temp_df.index = pd.to_datetime(temp_df.index, errors='coerce')
        temp_df = temp_df.rename_axis('snapped_at').reset_index()
        temp_df = temp_df.dropna(subset=['snapped_at'])
        temp_df['snapped_at'] = temp_df['snapped_at'].dt.tz_localize(None)

        # Sinkronkan tanggal actual dengan prediksi
        start_pred = temp_df['snapped_at'].min()
        end_pred = temp_df['snapped_at'].max()
        df_filtered_actual = df_actual[(df_actual['snapped_at'] >= start_pred) & (df_actual['snapped_at'] <= end_pred)]

        # Inisialisasi df_pred pertama kali
        if df_pred is None:
            df_pred = df_filtered_actual[['snapped_at', 'price']].copy()

        # Ambil kolom prediksi yang sesuai
        pred_col = [col for col in temp_df.columns if col.startswith("price_pred_")]
        if pred_col:
            col_pred = pred_col[0]
            df_pred = df_pred.merge(temp_df[['snapped_at', col_pred]], on='snapped_at', how='left')

            # Evaluasi model
            y_true = df_pred['price'].values
            y_pred = df_pred[col_pred].values

            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100
            r2 = r2_score(y_true, y_pred)

            eval_results[full_model_name] = {
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "R2": r2
            }

    # Visualisasi
    df_plot = df_pred.rename(columns={
        'price': 'Harga Aktual',
        'price_pred_xgb': 'Harga Prediksi XGBoost',
        'price_pred_rf': 'Harga Prediksi Random Forest',
        'price_pred_dtree': 'Harga Prediksi Decision Tree'
    })

    fig = px.line(df_plot, x='snapped_at',
                y=[col for col in df_plot.columns if col != 'snapped_at'],
                labels={'snapped_at': 'Tanggal', 'value': 'Harga (USD)', 'variable': 'Tipe'},
                title=f"Prediksi Harga {koin_pred} oleh Berbagai Model")
    fig.update_traces(mode='lines+markers')
    st.plotly_chart(fig, use_container_width=True)

    # Tampilkan evaluasi
    st.markdown("### 📊 Evaluasi Model")
    eval_df = pd.DataFrame(eval_results).T.reset_index().rename(columns={'index': 'Model'})
    eval_df = eval_df[['Model', 'MAE', 'RMSE', 'MAPE', 'R2']]
    eval_df[['MAE', 'RMSE']] = eval_df[['MAE', 'RMSE']].round(2)
    eval_df['MAPE'] = eval_df['MAPE'].round(2)
    eval_df['R2'] = eval_df['R2'].round(2)
    st.dataframe(eval_df)

    # Tabel data akhir
    st.markdown("### 📋 Data Prediksi")
    st.dataframe(df_plot)

# Visualisasi perbandingan
else:
    df_all = pd.concat([load_data("BTC"), load_data("ETH")], ignore_index=True)

    tab1, tab2, tab3 = st.tabs([
        "Perbandingan Harga", "Perbandingan Kapitalisasi Pasar", "Perbandingan Volume Perdagangan"
    ])

    with tab1:
        st.subheader("📈 Perbandingan Harga BTC vs ETH")
        filtered_df = filter_date(df_all, "harga")
        fig = px.line(filtered_df, x='snapped_at', y='price', color='koin', labels={'snapped_at': 'Tanggal', 'price': 'Harga (USD)', 'koin': 'Koin'})
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📈 Perbandingan Kapitalisasi Pasar")
        filtered_df = filter_date(df_all, "cap")
        fig = px.line(filtered_df, x='snapped_at', y='market_cap', color='koin', labels={'snapped_at': 'Tanggal', 'market_cap': 'Kapitalisasi Pasar (USD)', 'koin': 'Koin'})
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("📈 Perbandingan Volume Perdagangan")
        filtered_df = filter_date(df_all, "volume")
        fig = px.line(filtered_df, x='snapped_at', y='total_volume', color='koin', labels={'snapped_at': 'Tanggal', 'total_volume': 'Volume Perdagangan (USD)', 'koin': 'Koin'})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"Sumber: [CoinGecko](https://www.coingecko.com)")