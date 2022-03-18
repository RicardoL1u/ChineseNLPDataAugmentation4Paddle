from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForMaskedLM.from_pretrained('bert-base-chinese')

inputs = tokenizer("巴黎是[MASK]国的首都.", return_tensors="pt")
labels = tokenizer("巴黎是法国的首都.", return_tensors="pt")["input_ids"]

outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits
print(logits.shape)
predict_id = torch.argmax(logits.squeeze()[4]).unsqueeze(dim=0)
predict_token = tokenizer.convert_ids_to_tokens(predict_id)
print(predict_id,predict_token)