from transformers import BertForSequenceClassification, BertTokenizer
import torch

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', cache_dir='../result/', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_ids = tokenizer.encode("[CLS] Hello, how are you? [SEP]")
print(input_ids)
input_ids = torch.tensor([input_ids], dtype=torch.long)
print(input_ids)
inputs_embeds = model.bert.embeddings.word_embeddings(input_ids)

y = model(inputs_embeds=inputs_embeds)
print(y)
