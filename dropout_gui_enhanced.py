
"""dropout_gui_enhanced.py
A prettier, resizable Tkinter GUI for predicting dropout probability.
"""
import sys, os
from pathlib import Path

def resource_path(rel_path):
    # PyInstaller 會把 bundle 解到 _MEIPASS
    base = getattr(sys, "_MEIPASS", Path(__file__).parent)
    return Path(base) / rel_path

# 然後把 CSV_PATH / MODEL_PATH 這樣改：
CSV_PATH   = resource_path("JEE_Dropout_After_Class_12.csv")
MODEL_PATH = resource_path("dropout_model.joblib")

import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

CSV_PATH = Path("JEE_Dropout_After_Class_12.csv")
MODEL_PATH = Path("dropout_model.joblib")

# 加入 admission_taken
FEATURES = [
    "family_income",
    "parent_education",
    "peer_pressure_level",
    "admission_taken",
    "daily_study_hours"
]
CATEGORICAL = [
    "family_income",
    "parent_education",
    "peer_pressure_level",
    "admission_taken"
]
NUMERICAL = ["daily_study_hours"]


# --------------------------- ML helpers --------------------------------------
def build_pipeline():
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERICAL),
        ]
    )
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)

    return Pipeline([("pre", pre), ("lr", lr)])


def train_model():
    if not CSV_PATH.exists():
        messagebox.showerror("File not found", f"Dataset {CSV_PATH} not found.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    df["daily_study_hours"].fillna(df["daily_study_hours"].median(), inplace=True)
    X = df[FEATURES]
    y = df["dropout"].astype(int)

    pipe = build_pipeline()
    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_PATH)
    return pipe


def load_or_train():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else train_model()


# ----------------------------- GUI App ---------------------------------------
class PrettyApp(tk.Tk):
    BG = "#202A44"
    ACCENT = "#4F9D69"
    TEXT = "#FFFFFF"

    def __init__(self):
        super().__init__()
        self.title("JEE Dropout Predictor")
        self.configure(bg=self.BG)

        # allow resizing
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        for i in range(8):
            self.rowconfigure(i, weight=1)

        self.model = load_or_train()
        self._build_style()
        self._create_widgets()

    def _build_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", background=self.BG, foreground=self.TEXT, font=("Segoe UI", 11))
        style.configure("TButton", background=self.ACCENT, foreground=self.TEXT, font=("Segoe UI", 11, "bold"))
        style.map(
            "TButton",
            background=[("active", "#3E8357"), ("disabled", "#6c6c6c")],
            foreground=[("disabled", "#d0d0d0")],
        )
        style.configure("TCombobox", fieldbackground="#f0f0f0", padding=4)
        style.configure("TSpinbox", arrowsize=14)

    def _create_widgets(self):
        pad = {"padx": 12, "pady": 6, "sticky": "ew"}

        # Family income
        tk.Label(self, text="Family income").grid(row=0, column=0, **pad)
        self.family_income = ttk.Combobox(self, state="readonly")
        self.family_income["values"] = ("Low", "Medium", "High")
        self.family_income.current(0)
        self.family_income.grid(row=0, column=1, **pad)

        # Parent education
        tk.Label(self, text="Parent education").grid(row=1, column=0, **pad)
        self.parent_education = ttk.Combobox(self, state="readonly")
        self.parent_education["values"] = ("Upto 10th", "Graduate", "PG", "12th")
        self.parent_education.current(0)
        self.parent_education.grid(row=1, column=1, **pad)

        # Peer pressure level
        tk.Label(self, text="Peer pressure level").grid(row=2, column=0, **pad)
        self.peer_pressure = ttk.Combobox(self, state="readonly")
        self.peer_pressure["values"] = ("Low", "Medium", "High")
        self.peer_pressure.current(1)
        self.peer_pressure.grid(row=2, column=1, **pad)

        # Admission taken
        tk.Label(self, text="Admission taken").grid(row=3, column=0, **pad)
        self.admission_taken = ttk.Combobox(self, state="readonly")
        self.admission_taken["values"] = ("No", "Yes")
        self.admission_taken.current(0)
        self.admission_taken.grid(row=3, column=1, **pad)

        # Daily study hours
        tk.Label(self, text="Daily study hours").grid(row=4, column=0, **pad)
        self.study_hours = ttk.Spinbox(self, from_=0, to=12, increment=0.5, width=8)
        self.study_hours.set("4")
        self.study_hours.grid(row=4, column=1, **pad)

        # Predict button
        predict_btn = ttk.Button(self, text="Predict", command=self._predict)
        predict_btn.grid(row=5, column=0, columnspan=2, pady=(10, 2), sticky="nsew")

        # Result display
        self.result_var = tk.StringVar(value="Dropout Probability —")
        result_lbl = tk.Label(
            self,
            textvariable=self.result_var,
            font=("Segoe UI", 14, "bold"),
            bg=self.BG,
            fg=self.ACCENT
        )
        result_lbl.grid(row=6, column=0, columnspan=2, pady=(4, 12))

        # Ensure combobox/spinbox expand
        for child in self.winfo_children():
            if isinstance(child, (ttk.Combobox, ttk.Spinbox)):
                child.configure(width=18)

    def _predict(self):
        try:
            hrs = float(self.study_hours.get())
        except ValueError:
            messagebox.showwarning("Invalid input", "Daily study hours must be numeric.")
            return

        feats = {
            "family_income": self.family_income.get(),
            "parent_education": self.parent_education.get(),
            "peer_pressure_level": self.peer_pressure.get(),
            "admission_taken": self.admission_taken.get(),
            "daily_study_hours": hrs,
        }
        prob = self.model.predict_proba(pd.DataFrame([feats]))[0, 1]
        self.result_var.set(f"Dropout Probability: {prob:.1%}")


if __name__ == "__main__":
    PrettyApp().mainloop()

