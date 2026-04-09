import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

file_path = "Customer_Data.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    
    if "CUST_ID" in df.columns:
        df = df.drop(columns=["CUST_ID"])
    
    df = df.dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    k = 4 
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)

    joblib.dump(model, "kmeans_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Модель и scaler сохранены!")

else:
    print("Файл Customer_Data.csv не найден! Положи его рядом с train.py")
