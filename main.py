import torch
from transformers import AdamW,  BertModel
import numpy as np
from tqdm import tqdm 
import random
import argparse

from dataloader import get_all_dataloader
from model import BERT_Regression

import transformers

new_version = False
if transformers.__version__ == '2.2.2':
    new_version = True

if new_version:
    from transformers import get_linear_schedule_with_warmup
else:
    from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/raw')
parser.add_argument('--processed_dir', default='./data/processed')

parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def get_mse(model, dataloader):
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in tqdm(dataloader, desc="eval"):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():        
            preds = model(b_input_ids, b_input_mask)
        
        tmp_eval_mse = (preds - b_labels)**2).sum()
        eval_mse += tmp_eval_accuracy.item()

    return eval_mse/len(dataloader)


bert_model = BertModel.from_pretrained("bert-base-uncased")
model = BERT_Regression(bert_model, args.hidden_size, args.dropout)
model = model.to(device)

train_dataloader, validation_dataloader, test_dataloader = get_all_dataloader(args.data_dir, args.processed_dir, args.batch_size)

optimizer = AdamW(model.parameters(),lr = args.lr)
total_steps = len(train_dataloader) * args.epochs
if new_version:
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
else:
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                        warmup_steps = 0,
                                        t_total = total_steps)

best_val_mse = 1e10
test_mse = 0

for epoch_i in range(args.epochs):
    total_loss = 0
    model.train()
        
    # For each batch of training data...
    for step, batch in tqdm(enumerate(train_dataloader)):
        model.train()
        
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
                
        preds = model(b_input_ids,b_input_mask)
        
        loss = (preds - b_labels)**2).sum()

        loss.backward()
        total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        model.zero_grad()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)            
    
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    val_mse = get_mse(model, validation_dataloader)
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        test_mse = get_mse(model, test_dataloader)
    print(" Val mse {}, test mse {}".format(val_mse, test_mse))

print("")
print("Training complete!")
                                       

