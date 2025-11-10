import pickle

with open("models/preprocessing_objects.pkl", "rb") as f:
    obj = pickle.load(f)

print(obj)
