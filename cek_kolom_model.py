import pickle

with open("model_waktu_prediksi.pkl", "rb") as f:
    model = pickle.load(f)

try:
    print(model.feature_names_in_)
except:
    print("Model tidak punya attribute feature_names_in_.")
    print("Attribute yang tersedia: ", dir(model))
