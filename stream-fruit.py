import streamlit as st
import pickle
import pandas as pd

file = 'fruit.xlsx'
fruit = pd.read_excel(file)
x = fruit[['diameter', 'weight', 'red', 'green', 'blue']]
y = fruit['name']

label_to_class = {'grapefruit': 0, 'orange': 1}
class_to_label = {v: k for k, v in label_to_class.items()}

def load_model_and_scaler(model_file, scaler_file=None):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    scaler = None
    if scaler_file:
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
    return model, scaler

def predict_fruit(features, model, scaler=None):
    import numpy as np
    
    features = np.array(features).reshape(1, -1)

    if scaler:
        features = scaler.transform(features)
    
    prediction_class = model.predict(features)[0]
    prediction_label = class_to_label[prediction_class]
    return prediction_label, prediction_class


st.title("Aplikasi Prediksi Buah Dengan Model")
st.write("Pilih algoritma yang digunakan, masukkan fitur buah untuk memprediksi jenis buah.")

algorithm = st.selectbox("Pilih Algoritma", ["Random Forest", "SVM", "Perceptron"])

if algorithm == "Random Forest":
    model_file = 'fruit_RandomForest.pkl'
    model, scaler = load_model_and_scaler(model_file)
elif algorithm == "SVM":
    model_file = 'fruit_SVM.pkl'
    scaler_file = 'scaler_svm.pkl'
    model, scaler = load_model_and_scaler(model_file, scaler_file)
elif algorithm == "Perceptron":
    model_file = 'fruit_Perceptron.pkl'
    scaler_file = 'scaler_perceptron.pkl'
    model, scaler = load_model_and_scaler(model_file, scaler_file)

input_features = []
for col in x.columns:
    value = st.number_input(f"Masukkan nilai untuk {col}:", value=0.0)
    input_features.append(value)

if all(input_features):  # Pastikan semua input ada
    if st.button("Prediksi"):
        label, class_index = predict_fruit(input_features, model, scaler)
        st.success(f"Model memprediksi jenis buah: {label} (Cluster: {class_index})")
else:
    st.warning("Harap masukkan semua nilai fitur sebelum memprediksi.")
