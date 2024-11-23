from difflib import SequenceMatcher

# Kata kunci kategori
known_categories = {
    "makan": ["makan", "minum", "sarapan", "makan siang", "makan malam", "restoran", "warung", "cafe"], 
    "transportasi": ["transportasi", "kendaraan", "bensin", "taksi", "kereta", "bus", "bis", "ojek", "mobil", "motor"], 
    "hiburan": ["hiburan", "bioskop", "game", "film", "musik", "konser", "pesta", "healing", "jalan jalan", "main"], 
    "tagihan": ["tagihan", "listrik", "air", "internet", "kuliah", "wifi", "sekolah", "pendidikan", "pajak", "bpjs", "kesehatan", "berobat", "rumah sakit"], 
    "belanja": ["belanja", "shopping", "mall", "supermarket", "pasar", "online", "skincare", "elektronik", "make up", "kebutuhan"], 
    "lain lain": ["lain lain", "lainnya"]
}

def preprocess_text(text):
    """Preprocess input text to lower case and split into words."""
    return text.lower().split()

def get_similarity_score(word1, word2):
    """Calculate similarity score between two words."""
    return SequenceMatcher(None, word1, word2).ratio()

def map_to_category(user_input):
    """
    Map user input to the closest known category based on keyword similarity.
    """
    tokens = preprocess_text(user_input)
    category_scores = {category: 0 for category in known_categories}

    for token in tokens:
        for category, keywords in known_categories.items():
            # Calculate similarity for each token against keywords
            scores = [get_similarity_score(token, keyword) for keyword in keywords]
            max_score = max(scores) if scores else 0
            category_scores[category] += max_score

    # Return category with the highest total score
    best_category = max(category_scores, key=category_scores.get)
    print(f"Input: {user_input}, Best Match: {best_category}, Scores: {category_scores}")
    return best_category
