import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("heart_model_xgb.pkl")
scaler = joblib.load("scaler.pkl")

# Label mappings
mappings = {
    "Sex": {0: "Female", 1: "Male"},
    "Chest_Pain_Type": {
        0: "Typical Angina", 1: "Atypical Angina",
        2: "Non-anginal Pain", 3: "Asymptomatic"
    },
    "Resting_ECG": {
        0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"
    },
    "Exercise_Induced_Angina": {0: "No", 1: "Yes"},
    "ST_Slope": {0: "Up", 1: "Flat", 2: "Down"},
    "Thallium_Stress_Test": {
        0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Other"
    }
}

# Reverse mappings
reverse_mappings = {feat: {v: k for k, v in d.items()} for feat, d in mappings.items()}

# Feature layout: (name, type)
features = [
    ("Age", "numeric"),
    ("Sex", "categorical"),
    ("Chest_Pain_Type", "categorical"),
    ("Resting_BP", "numeric"),
    ("Cholesterol", "numeric"),
    ("Fasting_Blood_Sugar", "numeric"),
    ("Resting_ECG", "categorical"),
    ("Max_Heart_Rate", "numeric"),
    ("Exercise_Induced_Angina", "categorical"),
    ("Oldpeak", "numeric"),
    ("ST_Slope", "categorical"),
    ("Number_of_Vessels_Fluro", "numeric"),
    ("Thallium_Stress_Test", "categorical")
]

# Build GUI
root = tk.Tk()
root.title("Heart Disease Risk Predictor")
root.geometry("620x750")
root.configure(bg="white")

tk.Label(root, text="💓 Heart Disease Risk Predictor", font=("Arial", 20, "bold"), bg="white", fg="darkred").pack(pady=10)

form_frame = tk.Frame(root, bg="white")
form_frame.pack(pady=10)

entries = {}
dropdowns = {}

for idx, (feature, ftype) in enumerate(features):
    tk.Label(form_frame, text=feature, font=("Arial", 12), anchor="w", width=25, bg="white").grid(row=idx, column=0, pady=6, sticky="w")

    if ftype == "categorical":
        combo = ttk.Combobox(form_frame, values=list(mappings[feature].values()), state="readonly", width=30)
        combo.grid(row=idx, column=1, padx=10)
        dropdowns[feature] = combo
    else:
        entry = tk.Entry(form_frame, width=33)
        entry.grid(row=idx, column=1, padx=10)
        entries[feature] = entry

# Prediction logic
def predict():
    try:
        input_data = []
        for feature, ftype in features:
            if ftype == "categorical":
                value = dropdowns[feature].get()
                if value == "":
                    raise ValueError(f"Select a value for {feature}")
                input_data.append(reverse_mappings[feature][value])
            else:
                val = entries[feature].get()
                if val.strip() == "":
                    raise ValueError(f"Enter a value for {feature}")
                input_data.append(float(val))

        arr = np.array(input_data).reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        prediction = model.predict(arr_scaled)[0]
        probability = model.predict_proba(arr_scaled)[0][1]

        # Risk interpretation
        if probability >= 0.7:
            result = "🔴 High Risk"
        elif probability >= 0.4:
            result = "🟠 Borderline Risk"
        else:
            result = "🟢 Healthy"

        messagebox.showinfo("Prediction Result", f"Prediction: {result}")

    except Exception as e:
        messagebox.showerror("Input Error", str(e))

# Predict Button
tk.Button(root, text="Predict", font=("Arial", 14), bg="darkred", fg="white", width=20, command=predict).pack(pady=20)

# Disclaimer
tk.Label(root, text="Disclaimer: This is a prediction tool only.\nConsult a doctor for accurate diagnosis.",
         font=("Arial", 10), fg="gray", bg="white").pack(pady=10)

root.mainloop()
