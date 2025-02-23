import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from PIL import Image  # Import the PIL library
import requests
import html

st.set_page_config(page_title="Hitung IP Broiler dengan mudah", page_icon="🐔")

# Access your bot token and chat ID from Streamlit secrets
BOT_TOKEN = st.secrets["bot_token"]
CHAT_ID = st.secrets["chat_id"]

def send_file_to_telegram(file_path, bot_token, chat_id):
    url = f'https://api.telegram.org/bot{bot_token}/sendDocument'
    with open(file_path, 'rb') as file:
        response = requests.post(url, data={'chat_id': chat_id}, files={'document': file})
    return response

# Title of the Streamlit app
col1, col2 = st.columns([1, 2])  # Adjust the ratio as needed

with col1:
    try:
        image = Image.open("header.jpeg")
        st.image(image, use_container_width=True, width=120)  # Adjust width as needed
    except FileNotFoundError:
        st.warning("header.jpeg tidak ditemukan.")

with col2:
    st.title("Tool Menghitung Indeks Performans (IP) Broiler")

# Load data (you might need to adjust the path)
DATA_URL = 'data_kandang.csv'

@st.cache_data  # Use st.cache_data for caching data loading
def load_data(url):
    data = pd.read_csv(url, thousands=',')
    return data

data = load_data(DATA_URL)

# Define features and target
X = data[['Age', 'Total_Body_Weight', 'FCR', 'Live_Bird', 'Ayam_Dipelihara', 'persen_Live_Bird']]
y = data['IP']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# st.write(f"Mean Squared Error on Test Data: {mse:.2f}")
# st.write(f"R^2 Score on Test Data: {r2:.2f}")

# Custom CSS for green button
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #00FF00 !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# Streamlit input widgets for prediction
st.sidebar.header("Masukkan Parameter Produksi Broiler")
Age = st.sidebar.number_input("Umur Ayam (Hari)", min_value=1, value=0)
FCR = st.sidebar.number_input("FCR", min_value=0.0, value=0.0)
Ayam_Dipelihara = st.sidebar.number_input("Jumlah Ayam Dipelihara (ekor)", min_value=0, value=0)
persen_Live_Bird = st.sidebar.number_input("Persentase Ayam Hidup (%)", min_value=50.0, max_value=100.0, value=50.0)
Total_Body_Weight = st.sidebar.number_input("Total Berat Badan Panen (kg)", min_value=10.0, value=10.0)

# Predict button
if st.sidebar.button("Hitung Indeks Performans"):
    # Calculate Live_Bird from Ayam_Dipelihara and persen_Live_Bird
    Live_Bird = (persen_Live_Bird / 100) * Ayam_Dipelihara

    # Create input data for prediction
    input_data = pd.DataFrame([[Age, Total_Body_Weight, FCR, Live_Bird, Ayam_Dipelihara, persen_Live_Bird]],
                                columns=['Age', 'Total_Body_Weight', 'FCR', 'Live_Bird', 'Ayam_Dipelihara', 'persen_Live_Bird'])

    # Make prediction
    if Age == 0 and Total_Body_Weight == 0 and FCR == 0 and Live_Bird == 0 and Ayam_Dipelihara == 0 and persen_Live_Bird == 0:
        prediction = "Silahkan masukkan data produksi broiler Anda"
        actual_ip = "Silahkan masukkan data produksi broiler Anda"
    else:
        prediction = model.predict(input_data)[0]

    # Calculate actual IP
    if FCR > 0 and Age > 0 and Live_Bird > 0:
        actual_ip = ((persen_Live_Bird * (Total_Body_Weight/Live_Bird)) / (FCR * Age)) * 100
    else:
        actual_ip = "Silahkan masukkan data produksi broiler Anda"  # Avoid division by zero

    # Display prediction
    if isinstance(prediction, str):
        st.warning(f"IP Prediksi: {prediction}")
    else:
        st.success(f"IP Prediksi: {prediction:.2f}")

    if isinstance(actual_ip, str):
        st.warning(f"IP Aktual: {actual_ip}")
    else:
        st.success(f"IP Aktual: {actual_ip:.2f}")

    # Interpretasi IP Prediksi
    if isinstance(prediction, str):
        st.warning(f"Interpretasi IP Prediksi: {prediction}")
        interpretasi_prediksi = "Silahkan masukkan data produksi broiler Anda"
    else:
        if prediction < 300:
            interpretasi_prediksi = "Kurang"
        elif 301 <= prediction <= 325:
            interpretasi_prediksi = "Cukup"
        elif 326 <= prediction <= 350:
            interpretasi_prediksi = "Baik"
        elif 351 <= prediction <= 400:
            interpretasi_prediksi = "Sangat Baik"
        elif prediction > 500:
            interpretasi_prediksi = "Silahkan cek kembali data produksi broiler Anda"
        else:
            interpretasi_prediksi = "Istimewa"
        st.success(f"Interpretasi IP Prediksi: {interpretasi_prediksi}")

    # Interpretasi IP Aktual
    if isinstance(actual_ip, str):
        st.warning(f"Interpretasi IP Aktual: {actual_ip}")
    else:
        if actual_ip < 300:
            interpretasi = "Kurang"
        elif 301 <= actual_ip <= 325:
            interpretasi = "Cukup"
        elif 326 <= actual_ip <= 350:
            interpretasi = "Baik"
        elif 351 <= actual_ip <= 400:
            interpretasi = "Sangat Baik"
        elif 401 <= actual_ip <= 500:
            interpretasi = "Istimewa"
        else:
            interpretasi = "Silahkan cek kembali data produksi broiler Anda"
        st.success(f"Interpretasi IP Aktual: {interpretasi}")

    # Compare Interpretations
    if 'interpretasi' in locals() and 'interpretasi_prediksi' in locals():
        if interpretasi != interpretasi_prediksi:
            st.info("Perbedaan interpretasi antara IP Aktual dan IP Prediksi: System memerlukan lebih banyak data untuk meningkatkan akurasi dugaan.")
        else:
            # Save data to CSV if interpretations are the same
            culling = (Ayam_Dipelihara * (100 - persen_Live_Bird))/100
            today = datetime.date.today()
            ADG_actual = ((Total_Body_Weight/Live_Bird)*1000) if Live_Bird > 0 else 0  # Calculate ADG_actual, avoid division by zero
            feed = FCR * Total_Body_Weight
            new_data = pd.DataFrame([[Age, today, Total_Body_Weight, FCR, Live_Bird, Ayam_Dipelihara, persen_Live_Bird, actual_ip, prediction, culling, ADG_actual, feed]],
                                    columns=['Age', 'Date', 'Total_Body_Weight', 'FCR', 'Live_Bird', 'Ayam_Dipelihara', 'persen_Live_Bird', 'IP_actual', 'IP', 'Culling', 'ADG_actual', 'Feed'])
            try:
                existing_data = pd.read_csv('prediksi.csv')

                # Rename columns in existing data if they exist
                if 'actual_ip' in existing_data.columns:
                    existing_data.rename(columns={'actual_ip': 'IP_actual'}, inplace=True)
                if 'prediction' in existing_data.columns:
                    existing_data.rename(columns={'prediction': 'IP'}, inplace=True)
                # Rename FCR to FCR_actual
                if 'FCR' in existing_data.columns:
                    existing_data.rename(columns={'FCR': 'FCR_actual'}, inplace=True)

                # Define the desired column order
                column_order = ['Age', 'Date', 'Total_Body_Weight', 'FCR', 'Live_Bird', 'Ayam_Dipelihara', 'persen_Live_Bird', 'IP_actual', 'IP', 'Culling', 'ADG_actual', 'Feed']

                # Reorder the columns, handling missing columns
                existing_columns = existing_data.columns.tolist()
                ordered_columns = [col for col in column_order if col in existing_columns]
                missing_columns = [col for col in existing_columns if col not in ordered_columns]
                final_columns = ordered_columns + missing_columns  # Put existing columns first

                existing_data = existing_data[final_columns]

                # Ensure new_data has the same columns as existing_data before concatenating
                for col in existing_data.columns:
                    if col not in new_data.columns:
                        new_data[col] = np.nan  # Or some other appropriate default value

                new_data = new_data[existing_data.columns] #Ensure new_data has the same column order

                # Compare the new data with the last row of existing data
                if not existing_data.empty:
                    if not new_data.equals(existing_data.tail(1)):
                        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                        combined_data.to_csv('prediksi.csv', index=False)
                        file_has_changed = True
                    else:
                        file_has_changed = False
                else:
                    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                    combined_data.to_csv('prediksi.csv', index=False)
                    file_has_changed = True
            except FileNotFoundError:
                new_data.to_csv('prediksi.csv', index=False)
                file_has_changed = True
            
            # Send the file to Telegram bot
            if file_has_changed:
                response = send_file_to_telegram('prediksi.csv', BOT_TOKEN, CHAT_ID)
                if response.status_code == 200:
                    st.success("Terimakasih atas menggunakan system kami.")
                else:
                    st.error("System sedang sibuk, silahkan coba beberapa saat lagi.")
            else:
                st.info("Tidak ada data baru untuk dikirim ke Telegram.")
            st.success(f"Berikut data IP di kandang Anda, berdasarkan perhitungan maka nilainya {actual_ip} ({interpretasi}), dan berdasarkan prediksi dari system kami nilainya {prediction} ({interpretasi_prediksi})")
    else:
        st.info("System memerlukan lebih banyak data untuk meningkatkan akurasi prediksi.")

if st.sidebar.button("Hapus Data"):
    # Clear all Streamlit cache
    st.cache_data.clear()
    Age = 0
    FCR = 0.0
    Ayam_Dipelihara = 0
    persen_Live_Bird = 50.0
    Total_Body_Weight = 10.0
    st.rerun()

# Run the Streamlit app
if __name__ == "__main__":
    today = datetime.date.today()
    st.write(f"-----------------------------------------------------------------------------------------------------------------")
    st.write(f"Keterangan:")
    st.write("1. IP (Indeks Performans) adalah nilai yang menggambarkan performa produksi broiler.")
    st.write("2. Masukkan data produksi broiler Anda pada sidebar, dan system juga akan memberikan interpretasi IP yang dihasilkan, sekaligus memberikan prediksi IP berdasarkan data yang Anda masukkan, saat Anda klik tombol [Hitung Indeks Performans].")
    st.write("Data prediksi akan semakin presisi jika Anda sering menggunakan system ini untuk menghitung IP Broiler | alogaritma IP prediksi telah diupdate pada tanggal", today.strftime('%d %B %Y'))
# Footer
st.markdown("---")
current_year = datetime.datetime.now().year
st.text(f"© {current_year} Developed by: Galuh Adi Insani with ❤️. All rights reserved.")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)