import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

st.set_page_config(page_title="Сегментация клиентов", layout="wide")
st.title("Сегментация клиентов (KMeans)")

file_path = "Customer_Data.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    st.subheader("Исходные данные")
    st.dataframe(df)

    if "CUST_ID" in df.columns:
        ids = df["CUST_ID"]
        df = df.drop(columns=["CUST_ID"])
    else:
        ids = None

    df = df.dropna()
    
    k = st.slider("Выбери количество кластеров", 2, 10, 4)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)

    clusters = model.predict(X_scaled)
    df["Cluster"] = clusters
    if ids is not None:
        df["CUST_ID"] = ids

    st.subheader("Данные с кластерами")
    st.dataframe(df)

    st.subheader("Количество клиентов по кластерам")
    st.bar_chart(df["Cluster"].value_counts())

    st.subheader("Визуализация кластеров (PCA)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    st.pyplot(fig)

    if st.button("Сохранить модель"):
        joblib.dump(model, "kmeans_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        st.success("Модель и scaler успешно сохранены!")

else:
    st.error("Файл Customer_Data.csv не найден! Положи его рядом с app.py")
