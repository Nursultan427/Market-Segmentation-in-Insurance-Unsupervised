import pandas as pd
import joblib
import os

file_path = "Customer_Data_test.csv"  

if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    if "CUST_ID" in df.columns:
        ids = df["CUST_ID"]
        df = df.drop(columns=["CUST_ID"])
    else:
        ids = None

    df = df.dropna()

    if os.path.exists("kmeans_model.pkl") and os.path.exists("scaler.pkl"):
        model = joblib.load("kmeans_model.pkl")
        scaler = joblib.load("scaler.pkl")

        X_scaled = scaler.transform(df)
        clusters = model.predict(X_scaled)

        df["Cluster"] = clusters
        if ids is not None:
            df["CUST_ID"] = ids

        print("Результаты кластеризации:")
        print(df)
    else:
        print("Модель или scaler не найдены! Сначала запусти train.py")

else:
    print("Файл Customer_Data_test.csv не найден!")
