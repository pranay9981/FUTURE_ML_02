import torch
import json
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from dataset import TicketDataset # Reusing the cleaning logic

# Load Model & Config
MODEL_PATH = "saved_model"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
with open(f"{MODEL_PATH}/label_map.json", "r") as f:
    label_map = json.load(f)
    # Convert keys back to integers because JSON saves them as strings
    label_map = {int(k): v for k, v in label_map.items()}

cleaner = TicketDataset(tokenizer) # Dummy init just to access clean_text

def predict(text):
    # 1. Clean Text
    clean_text = cleaner.clean_text(text)
    
    # 2. Tokenize
    inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    
    # 3. Predict
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Get ID and Confidence
    probs = torch.nn.functional.softmax(logits, dim=1)
    pred_id = torch.argmax(probs).item()
    confidence = probs[0][pred_id].item()
    
    priority = assign_synthetic_priority(text)
    return label_map[pred_id], confidence, priority

def assign_synthetic_priority(text):
    """
    Rule-based priority assignment.
    """
    text = text.lower()
    if any(x in text for x in ['urgent', 'critical', 'immediately', 'outage', 'failure', 'deadline']):
        return 'High'
    elif any(x in text for x in ['error', 'issue', 'warning', 'slow', 'problem', 'unable']):
        return 'Medium'
    else:
        return 'Low'

if __name__ == "__main__":
    print("🤖 Ticket Classifier Ready. Type 'quit' to exit.")
    while True:
        user_input = input("\nEnter ticket text: ")
        if user_input.lower() == 'quit': break
        
        category, score, priority = predict(user_input)
        print(f"Category: {category} (Confidence: {score:.2%})")
        print(f"Priority: {priority}")