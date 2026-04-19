import torch
from transformers import BertTokenizer
from model import JargonFakeNewsModel

# 1. Initialize the model architecture
model = JargonFakeNewsModel()

# 2. Load the weights 
# Note: Users must download 'jargon_fake_news_model.pth' from Hugging Face first
try:
    model.load_state_dict(torch.load('jargon_fake_news_model.pth', map_location='cpu'), strict=False)
    model.eval()
    print("✅ Model weights loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'jargon_fake_news_model.pth' not found. Please download it from Hugging Face.")
    exit()

# 3. Setup tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_news(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        probability = model(inputs['input_ids'], inputs['attention_mask']).item()
    
    label = "FAKE" if probability > 0.5 else "REAL"
    confidence = probability if label == "FAKE" else 1 - probability
    
    print(f"\n--- Result ---")
    print(f"Headline: {text}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    print("Jargon Fake News Detector")
    user_text = input("Enter a headline to test: ")
    predict_news(user_text)
