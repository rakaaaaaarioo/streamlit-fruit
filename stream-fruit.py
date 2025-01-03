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
    if scaler:
        features = scaler.transform([features])
    prediction_class = model.predict([features])[0]  # Prediksi kelas
    prediction_label = class_to_label[prediction_class]  # Mapping ke label
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

if st.button("Prediksi"):
    label, class_index = predict_fruit(input_features, model, scaler)
    st.success(f"Model memprediksi jenis buah: {label} (Cluster: {class_index})")




















# with open('fruit_RandomForest.pkl', 'rb') as f:
#     rf_model = pickle.load(f)

# with open('fruit_Perceptron.pkl', 'rb') as f:
#     svm_model = pickle.load(f)

# with open('fruit_SVM.pkl', 'rb') as f:
#     perceptron_model = pickle.load(f)

# with open('fruit_KMeans.pkl', 'rb') as f:
#     kmeans_model = pickle.load(f)

# st.title("Aplikasi Prediksi Model")

# # Input untuk model klasifikasi
# feature1 = st.number_input("Random Forest Clasifier")
# feature2 = st.number_input("Perceptron")
# feature3 = st.number_input("SVM")
# feature4 = st.number_input("Kmeans")
# # Tambahkan input sesuai kebutuhan model Anda

# # Tombol untuk melakukan prediksi
# if st.button("Prediksi"):
#     # Lakukan prediksi menggunakan model yang dipilih
#     prediction_rf = rf_model.predict([[feature1, feature2]])
#     st.write(f"Prediksi Random Forest: {prediction_rf}")

#     prediction_svm = svm_model.predict([[feature1, feature2]])
#     st.write(f"Prediksi SVM: {prediction_svm}")

#     prediction_perceptron = perceptron_model.predict([[feature1, feature2]])
#     st.write(f"Prediksi Perceptron: {prediction_perceptron}")

#     # Untuk K-Means, Anda mungkin ingin menampilkan cluster
#     cluster = kmeans_model.predict([[feature1, feature2]])
#     st.write(f"Cluster K-Means: {cluster}")