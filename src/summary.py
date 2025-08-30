"""
model_summary.py - Display architecture of the trained wafer defect model
"""

from tensorflow.keras.models import load_model #type: ignore


if __name__ == "__main__":
    model = load_model("models/wafer_defect_model_generator.h5")
    model.summary()
