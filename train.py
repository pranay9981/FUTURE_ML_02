import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW  # <--- FIX: Imported from torch instead of transformers
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from dataset import TicketDataset
import os

# --- SETTINGS ---
FILE_PATH = "all_tickets_processed_improved_v3.csv"
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
SAVE_DIR = "saved_model"

def train():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")

    # 2. Prepare Data
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Load dataset
    full_dataset = TicketDataset(tokenizer, file_path=FILE_PATH)
    
    # Split 80% Train, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    # 3. Handle Class Imbalance
    # Calculate weights so the model pays attention to small categories
    all_labels = full_dataset.data['label_id'].values
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("⚖️  Class Weights Applied.")

    # 4. Initialize Model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(full_dataset.label_map)
    )
    model.to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Loss Function with Class Weights
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)

    # 5. Training Loop
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, attention_mask=mask)
            
            # Calculate loss manually to ensure weights are applied
            loss = loss_fn(outputs.logits, labels)
            
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"\rBatch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}", end="")
        
        print(f"\nAverage Loss: {total_loss / len(train_loader):.4f}")

    # 6. Evaluation
    print("\n📊 Evaluating Performance...")
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Print Report
    names = [full_dataset.label_map[i] for i in sorted(full_dataset.label_map.keys())]
    print(classification_report(true_labels, predictions, target_names=names))

    # 7. Save Model
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    
    # Save the label map explicitly for prediction later
    import json
    with open(f"{SAVE_DIR}/label_map.json", "w") as f:
        # Convert numpy int64 to regular int for JSON serialization
        clean_map = {int(k): v for k, v in full_dataset.label_map.items()}
        json.dump(clean_map, f)
        
    print(f"✅ Model saved to {SAVE_DIR}/")

if __name__ == "__main__":
    train()