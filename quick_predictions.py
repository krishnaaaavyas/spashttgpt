from load_model import load_trained_model, predict_single

# Load the model once
model, tokenizer, label_mapping, config, device = load_trained_model()

def quick_predict(text):
    """Quick prediction function"""
    predicted_label, confidence, _ = predict_single(
        model, tokenizer, text, label_mapping, config['max_len'], device
    )
    return predicted_label, confidence

# Usage examples
if __name__ == "__main__":
    # Test cases
    test_cases = [
        "Patient presents with acute abdominal pain",
        "Routine checkup, patient healthy",
        "Patient asking for specific drug by name",
        "Emergency case with multiple injuries"
    ]
    
    print("Quick Predictions:")
    print("="*50)
    
    for text in test_cases:
        label, conf = quick_predict(text)
        print(f"Text: {text}")
        print(f"→ {label} ({conf:.3f})\n")
