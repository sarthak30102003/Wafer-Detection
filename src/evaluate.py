"""
evaluate.py - Evaluate trained wafer defect model on test dataset
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.preprocessing.image import img_to_array #type: ignore


def main():
    model = load_model("models/wafer_defect_model_generator.h5")

    class_names = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc',
                   'Near-full', 'none', 'Random', 'Scratch']
    test_root = "data/test"

    total_images, correct_predictions = 0, 0
    true_labels, predicted_labels = [], []

    print(f"{'Image':<20} {'Predicted':<15} {'Inference'}")

    for class_folder in class_names:
        class_path = os.path.join(test_root, class_folder)
        if not os.path.isdir(class_path):
            print(f"⚠️ Warning: Folder not found: {class_path}")
            continue

        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Could not read image: {img_path}")
                continue

            img = cv2.resize(img, (128, 128))
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img, verbose=0)
            predicted_index = np.argmax(prediction[0])
            predicted_class = class_names[predicted_index]

            true_index = class_names.index(class_folder)
            true_labels.append(true_index)
            predicted_labels.append(predicted_index)

            is_correct = predicted_class == class_folder
            mark = "✅" if is_correct else "❌"
            print(f"{img_file:<20} {predicted_class:<15} {mark}")

            if is_correct:
                correct_predictions += 1
            total_images += 1

    # Accuracy
    accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
    print(f"\n✅ {correct_predictions}/{total_images} images predicted correctly "
          f"({accuracy:.2f}% accuracy)")

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Accuracy: {accuracy:.2f}%")
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix.png")
    plt.show()
    print("📊 Confusion matrix saved at results/confusion_matrix.png")


if __name__ == "__main__":
    main()
