import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from keras.models import load_model
from datetime import date
import os

# --- KONFIGURASI FILE (GANTI NAMA FILE DI SINI) ---
# Pastikan nama file ini SAMA PERSIS dengan file yang ada di folder Anda
MODEL_PATH = 'stock_model.h5'   # <--- Ubah nama file .h5 Anda di sini
SCALER_PATH = 'stock_scaler.pkl'      # <--- Ubah nama file .pkl Anda di sini

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Aplikasi Prediksi Harga Saham (Auto Load Model)")
st.write(f"Model dimuat otomatis dari: `{MODEL_PATH}` dan `{SCALER_PATH}`")

# --- 0. INISIALISASI SESSION STATE ---
if 'stock_data' not in st.session_state:
    st.session_state['stock_data'] = None

# --- 1. INPUT DATA (SIDEBAR) ---
st.sidebar.header("1. Pengaturan Data Saham")
data_source = st.sidebar.radio("Pilih Sumber Data:", ("Yahoo Finance", "Upload CSV"))

# OPSI A: Mengambil dari Yahoo Finance
if data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Masukkan Ticker Saham (contoh: NVDA, BBCA.JK):", "NVDA")
    start_date = st.sidebar.date_input("Tanggal Mulai:", date(2018, 1, 1))
    end_date = st.sidebar.date_input("Tanggal Akhir:", date.today())
    
    if st.sidebar.button("Ambil Data"):
        try:
            with st.spinner(f'Mengambil data {ticker} dari Yahoo Finance...'):
                data = yf.download(ticker, start=start_date, end=end_date)
                
                if len(data) > 0:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    data.reset_index(inplace=True)
                    
                    if 'Close' in data.columns:
                        st.session_state['stock_data'] = data[['Date', 'Close']]
                        st.success(f"Data {ticker} berhasil dimuat! ({len(data)} baris)")
                    else:
                        st.error("Kolom 'Close' tidak ditemukan.")
                else:
                    st.error("Data tidak ditemukan atau kosong.")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

# OPSI B: Upload CSV Manual (Untuk Data Saham saja)
elif data_source == "Upload CSV":
    uploaded_csv = st.sidebar.file_uploader("Upload File CSV Data Saham", type=['csv'])
    if uploaded_csv is not None:
        try:
            df_upload = pd.read_csv(uploaded_csv)
            df_upload.columns = [c.capitalize() for c in df_upload.columns]
            if 'Date' in df_upload.columns and 'Close' in df_upload.columns:
                df_upload['Date'] = pd.to_datetime(df_upload['Date'])
                df_upload.sort_values(by='Date', inplace=True)
                st.session_state['stock_data'] = df_upload
            else:
                st.error("File CSV harus memiliki kolom 'Date' dan 'Close'.")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

df = st.session_state['stock_data']

if df is not None:
    # Tampilkan Data Sekilas
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Grafik Data Historis")
        st.line_chart(df.set_index('Date')['Close'])
    with col2:
        st.subheader("Data Terakhir")
        st.write(df.tail(10))
        if st.button("Hapus Data / Reset"):
            st.session_state['stock_data'] = None
            st.rerun()

    # --- 2. KONFIGURASI PREDIKSI ---
    st.sidebar.markdown("---")
    st.sidebar.header("2. Parameter Prediksi")
    
    # PENTING: User harus memasukkan window yang SAMA dengan saat training
    prediction_days = st.sidebar.number_input("Window Hari (Harus sama dgn saat training):", 10, 100, 60)
    future_days = st.sidebar.slider("Prediksi Hari ke Depan:", 1, 60, 30)
    split_ratio = st.sidebar.slider("Rasio Pembanding (Train/Test View) %:", 60, 95, 80) / 100.0

    # --- 3. EKSEKUSI MODEL (AUTO LOAD) ---
    if st.sidebar.button("Jalankan Prediksi"):
        
        # Cek apakah file model dan scaler ada di folder
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            st.error(f"❌ File tidak ditemukan!\nPastikan file '{MODEL_PATH}' dan '{SCALER_PATH}' ada di satu folder dengan script ini.")
        else:
            try:
                with st.spinner('Memuat Model dari Local Storage & Menganalisis...'):
                    
                    # 1. LOAD MODEL & SCALER LANGSUNG DARI FILE LOKAL
                    model = load_model(MODEL_PATH)
                    scaler = joblib.load(SCALER_PATH)

                    # 2. PERSIAPAN DATA (PREPROCESSING)
                    data_close = df[['Close']].values 
                    
                    # Gunakan scaler yang di-load
                    scaled_data = scaler.transform(data_close)

                    # 3. SPLITTING DATA
                    training_data_len = int(len(scaled_data) * split_ratio)
                    test_data = scaled_data[training_data_len - prediction_days: , :]
                    
                    x_test = []
                    y_test = data_close[training_data_len:, :] # Harga Asli
                    
                    if len(test_data) <= prediction_days:
                        st.error("Data saham terlalu sedikit untuk Window Hari yang dipilih.")
                        st.stop()

                    for i in range(prediction_days, len(test_data)):
                        x_test.append(test_data[i-prediction_days:i, 0])
                        
                    x_test = np.array(x_test)
                    
                    try:
                        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
                    except ValueError:
                        st.error(f"❌ Error Bentuk Data. Pastikan 'Window Hari' ({prediction_days}) sama persis dengan yang digunakan model!")
                        st.stop()

                    # 4. PREDIKSI (VALIDASI)
                    predictions = model.predict(x_test, verbose=0)
                    predictions = scaler.inverse_transform(predictions)

                    # 5. MENGHITUNG ERROR (RMSE & MAPE)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    mape = mean_absolute_percentage_error(y_test, predictions) * 100
                    
                    st.markdown("### 📊 Hasil Evaluasi Model")
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric("RMSE (Rata-rata Meleset)", f"{rmse:.2f}")
                    col_m2.metric("MAPE (Error Persentase)", f"{mape:.2f}%")

                    # 6. PLOT HASIL
                    st.subheader("Validasi: Harga Asli vs Prediksi Model")
                    fig1, ax1 = plt.subplots(figsize=(12, 6))
                    
                    valid = df[training_data_len:]
                    valid['Predictions'] = predictions
                    
                    ax1.plot(df['Date'][:training_data_len], df['Close'][:training_data_len], label='Data Training', alpha=0.5)
                    ax1.plot(valid['Date'], valid['Close'], label='Harga Asli', color='blue')
                    ax1.plot(valid['Date'], valid['Predictions'], label='Prediksi AI', color='red')
                    
                    ax1.set_xlabel('Tanggal')
                    ax1.set_ylabel('Harga')
                    ax1.legend()
                    st.pyplot(fig1)

                    # 7. FORECASTING MASA DEPAN
                    st.markdown("---")
                    st.subheader(f"🔮 Prediksi {future_days} Hari ke Depan")
                    
                    last_days_scaled = scaled_data[-prediction_days:]
                    current_batch = last_days_scaled.reshape(1, prediction_days, 1)
                    future_predictions = []
                    
                    for _ in range(future_days):
                        current_pred = model.predict(current_batch, verbose=0)[0]
                        future_predictions.append(current_pred)
                        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

                    future_predictions = scaler.inverse_transform(future_predictions)

                    last_date = df['Date'].iloc[-1]
                    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days)
                    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions.flatten()})

                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    ax2.plot(df['Date'].tail(100), df['Close'].tail(100), label='Harga Terakhir')
                    ax2.plot(forecast_df['Date'], forecast_df['Predicted Price'], label='Prediksi Masa Depan', color='green', linestyle='--')
                    ax2.legend()
                    st.pyplot(fig2)
                    
                    st.write("Tabel Prediksi:")
                    st.dataframe(forecast_df)

            except Exception as e:
                 st.error(f"Terjadi error saat menjalankan model: {e}")

else:
    st.info(f"Silakan 'Ambil Data' di sidebar. Model akan dimuat otomatis dari: {MODEL_PATH}")