import re
import urllib.parse
import pandas as pd
import numpy as np
import joblib  # Using joblib instead of pickle

def extract_features(url):
    """
    Extract features from a given URL.
    Returns a dictionary of extracted features in the correct order.
    """
    parsed_url = urllib.parse.urlparse(url)

    # Extracting URL structure-based features
    url_length = len(url)
    n_dots = url.count(".")
    n_hypens = url.count("-")
    n_underline = url.count("_")
    n_slash = url.count("/")
    n_questionmark = url.count("?")
    n_equal = url.count("=")
    n_at = url.count("@")
    n_and = url.count("&")
    n_exclamation = url.count("!")
    n_space = url.count(" ")
    n_tilde = url.count("~")
    n_comma = url.count(",")
    n_plus = url.count("+")
    n_asterisk = url.count("*")
    n_hastag = url.count("#")
    n_dollar = url.count("$")
    n_percent = url.count("%")

    # Redirection Feature
    n_redirection = 1 if "//" in url else 0

    # Ordered feature dictionary (must match training data format)
    features = {
        "url_length": url_length,
        "n_dots": n_dots,
        "n_hypens": n_hypens,
        "n_underline": n_underline,
        "n_slash": n_slash,
        "n_questionmark": n_questionmark,
        "n_equal": n_equal,
        "n_at": n_at,
        "n_and": n_and,
        "n_exclamation": n_exclamation,
        "n_space": n_space,
        "n_tilde": n_tilde,
        "n_comma": n_comma,
        "n_plus": n_plus,
        "n_asterisk": n_asterisk,
        "n_hastag": n_hastag,
        "n_dollar": n_dollar,
        "n_percent": n_percent,
        "n_redirection": n_redirection
    }

    return features

def predict_url(url, model_path="Decision_Tree.joblib", scaler_path="scaler.joblib"):
    """
    Predict if a given URL is phishing or legitimate using the trained model.
    Uses joblib to load the model and scaler.
    """
    # Extract features from the URL
    features = extract_features(url)
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features])

    # Load the scaler
    scaler = joblib.load(scaler_path)

    # Standardize features
    features_scaled = scaler.transform(features_df)

    # Load the trained model
    model = joblib.load(model_path)

    # Make prediction
    prediction = model.predict(features_scaled)
    
    return "Phishing" if prediction[0] == 1 else "Legitimate"

if __name__ == "__main__":
    # Example usage
    test_url = input("Enter a URL to check: ")
    result = predict_url(test_url)
    print(f"Prediction: {result}")
