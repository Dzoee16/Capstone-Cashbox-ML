import pandas as pd
import numpy as np

# Mapping kategori ke label urgensi
# 0=tidak penting, 1=sedang/rutin, 2=penting, 3=sangat mendesak
category_to_urgency = {
    "makan": 1,
    "transportasi": 2,
    "hiburan": 0,
    "tagihan": 3,
    "belanja": 2,
    "lain lain": 1
}

def preprocess_input(expenses):
    categories = ["makan", "transportasi", "hiburan", "tagihan", "belanja", "lain lain"]
    data = []

    for category, nominal in expenses:
        # One-hot encode category (pastikan semua kategori ada)
        category_one_hot = [1 if c == category else 0 for c in categories]
        # Normalize nominal
        # Tambahkan urgensi kategori
        normalized_nominal = nominal / 1e6  # Assuming nominal max is in millions
        urgency_score = category_to_urgency.get(category, 0)
        data.append(category_one_hot + [normalized_nominal, urgency_score])

    return np.array(data)

def preprocess_training_data(filepath):
    """
    Load and preprocess training data from CSV file.
    This function processes the data by mapping categories to urgency,
    one-hot encoding categories, and normalizing the nominal values.
    """
    # Load the data from the given CSV file
    df = pd.read_csv(filepath)
    
    # 1. Mengubah Time menjadi datetime
    df['Time'] = pd.to_datetime(df['Time'])
    
    # 2. Mengonversi Income/Expense menjadi numerik (1 untuk 'income' dan 0 untuk 'expense')
    income_expense_mapping = {"income": 1, "expense": 0}
    df['Income/Expense'] = df['Income/Expense'].map(income_expense_mapping)

    # 3. Mapping kategori ke label urgensi (menggunakan dictionary 'category_to_urgency')
    urgency_labels = df['Category'].map(category_to_urgency).fillna(0).astype(int).values

    # 4. Mengonversi Category menjadi one-hot encoding
    category_one_hot = pd.get_dummies(df['Category'])

    # Pastikan hanya 6 kategori yang digunakan
    if category_one_hot.shape[1] > 6:
        category_one_hot = category_one_hot.iloc[:, :6]

    # 5. Menormalkan kolom Nominal
    df['Nominal'] = df['Nominal'] / 1e6  # Menormalkan nominal dalam juta

    # 6. Gabungkan data hasil one-hot encoding dan kolom Nominal
    X = pd.concat([df[['Income/Expense']], category_one_hot, df['Nominal']], axis=1)
    
    # Pastikan hanya ada 7 kolom
    X = X.iloc[:, :8]

    # Pastikan semua data numerik dalam X memiliki tipe yang sesuai
    X = X.astype(np.float32)  # Pastikan X adalah array numerik bertipe float32
    
    return X.values, urgency_labels


