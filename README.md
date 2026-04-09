# Сегментация клиентов (KMeans)

Этот проект выполняет сегментацию клиентов с помощью алгоритма KMeans и визуализацию результатов через Streamlit.

## 📂 Файлы проекта

- `train.py` — обучает модель KMeans на данных из `Customer_Data.csv` и сохраняет модель и scaler.
- `test.py` — применяет сохранённую модель к новым данным (`Customer_Data_test.csv`) и показывает результаты кластеризации.
- `app.py` — Streamlit-приложение для интерактивного анализа, визуализации и сохранения модели.
- `Customer_Data.csv` — исходные данные клиентов.
- `kmeans_model.pkl` — сохранённая модель KMeans (создаётся автоматически).
- `scaler.pkl` — сохранённый StandardScaler (создаётся автоматически).

## 🚀 Установка

1. Клонируй репозиторий:
```bash
git clone https://github.com/Nursultan427/Market-Segmentation-in-Insurance-Unsupervised.git
cd Market-Segmentation-in-Insurance-Unsupervised  

 КАК ЗАПУСКАТЬ STREAMLIT:
python -m streamlit run app.py(вместо app.py пишите название своей папки через которую запускаете свой проект в streamit)

БИБЛИОТЕКИ КОТОРЫЕ Я ИСПОЛЬЗОВАЛ:
pandas
scikit-learn
matplotlib
joblib
streamlit
