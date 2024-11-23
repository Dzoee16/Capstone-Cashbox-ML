import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import json
import tensorflow as tf
from src.similarity import map_to_category
from src.data_processing import preprocess_input

# Load TensorFlow model
model_path = "models/urgency_model.h5"
model = tf.keras.models.load_model(model_path)

# Map skor ke skala 0-3
def map_score_to_category(score):
    """Map score (0-1) to urgency level (0-3)."""
    if score < 0.25:
        return 0  # Tidak penting
    elif score < 0.5:
        return 1  # Sedang/Rutin
    elif score < 0.75:
        return 2  # Penting
    else:
        return 3  # Sangat Mendesak

def collect_expense_input():
    """Collect user input for expenses."""
    expenses = []
    while True:
        category_input = input("Masukkan kategori pengeluaran: ")
        nominal_input = float(input("Masukkan nominal pengeluaran: "))
        mapped_category = map_to_category(category_input)
        expenses.append((category_input, nominal_input))

        more = input("Tambah pengeluaran lain? (y/n): ").lower()
        if more != 'y':
            break
    return expenses

def main():
    print("===== Prediksi Urgensi Pengeluaran =====")

    # Collect expense inputs
    expenses = collect_expense_input()
    print(f"Pengeluaran yang dimasukkan: {json.dumps(expenses, indent=2)}")

    # Prepare data for prediction
    data = preprocess_input(expenses)  # Pastikan input sesuai dengan model
    predictions = model.predict(data)

    # Combine predictions with categories, scores, and labels
    results = []
    for (category, nominal), pred in zip(expenses, predictions):
        score = pred[0]
        urgency_level = map_score_to_category(score)
        urgency_labels = ["Tidak Penting", "Sedang/Rutin", "Penting", "Sangat Mendesak"]
        urgency_label = urgency_labels[urgency_level]
        results.append((category, nominal, score, urgency_level, urgency_label))

    # Sort results by urgency level (descending)
    results.sort(key=lambda x: x[2], reverse=True)

    # Display sorted results
    print("===== Hasil Prediksi =====")
    print(f"{'No.':<4}{'Kategori':<30}{'Nominal':<15}{'Skor':<10}{'Level':<10}{'Label':<20}")
    for i, (category, nominal, score, urgency_level, urgency_label) in enumerate(results, start=1):
        print(f"{i:<4}{category:<30}Rp{nominal:<15,.0f}{score:<10.4f}{urgency_level:<10}{urgency_label:<20}")

if __name__ == "__main__":
    main()
