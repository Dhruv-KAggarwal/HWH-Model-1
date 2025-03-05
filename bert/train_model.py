import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load dataset
df = pd.read_csv('e:/Hackathon/HWH/bert/synthetic_dna_beauty_dataset.csv')

# Preprocess dataset
df['text'] = df['Skin_Type'] + ' ' + df['Hair_Type'] + ' ' + df['Hair_Thickness'] + ' ' + df['Hair_Loss_Risk'] + ' ' + df['Acne_Risk'] + ' ' + df['Wrinkle_Risk'] + ' ' + df['Vitamin_D_Deficiency'] + ' ' + df['Iron_Deficiency'] + ' ' + df['Lactose_Intolerance'] + ' ' + df['Allergy_Risk_Fragrance'] + ' ' + df['Allergy_Risk_Sulfates'] + ' ' + df['Allergy_Risk_Parabens'] + ' ' + df['Chemicals_to_Avoid'] + ' ' + df['UV_Sensitivity'] + ' ' + df['Best_Skincare_Product'] + ' ' + df['Skincare_Ingredients'] + ' ' + df['Best_Haircare_Product'] + ' ' + df['Haircare_Ingredients'] + ' ' + df['Best_Supplement']
df['label'] = df['ID'].astype('category').cat.codes

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=0.2)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['label'].unique()))

# Tokenize dataset
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

class DnaDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = DnaDataset(train_encodings, train_labels)
val_dataset = DnaDataset(val_encodings, val_labels)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()

# Save model
model.save_pretrained('e:/Hackathon/HWH/bert/model')
tokenizer.save_pretrained('e:/Hackathon/HWH/bert/model')
