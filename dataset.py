import pandas as pd
import re
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class TicketDataset(Dataset):
    def __init__(self, tokenizer, file_path="all_tickets_processed_improved_v3.csv", max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Load and Clean Data
        self.data, self.label_map = self.load_and_clean_data(file_path)
        print(f"✅ Data Loaded: {len(self.data)} tickets.")
        print(f"🏷️  Classes Found: {self.label_map}")

    def load_and_clean_data(self, path):
        df = pd.read_csv(path)
        
        # 1. Select specific columns
        df = df[['Document', 'Topic_group']].dropna()
        df.columns = ['text', 'label']

        # 2. Clean Text
        df['text'] = df['text'].apply(self.clean_text)

        # 3. Encode Labels
        le = LabelEncoder()
        df['label_id'] = le.fit_transform(df['label'])
        
        # Create map
        label_map = dict(zip(le.transform(le.classes_), le.classes_))
        
        return df, label_map

    def clean_text(self, text):
        if not isinstance(text, str): return ""
        text = text.lower()
        # Remove email headers and signatures
        text = re.sub(r'(from|sent|to|subject):.*', '', text)
        text = re.sub(r'(hi |hello |dear |best regards|thanks|kind regards).*', '', text)
        text = re.sub(r'ext \d+', '', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.strip()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.iloc[index].text)
        label = self.data.iloc[index].label_id

        # --- FIX: Calling tokenizer directly (Modern Way) ---
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }