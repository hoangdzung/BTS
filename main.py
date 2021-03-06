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
parser.add_argument('--datafile')
parser.add_argument('--dictfile')

parser.add_argument('--n_highway', type=int, default=4)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--lr2', type=float, default=0.01)
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

def get_mse(model, dataloader, scaler):
    model.eval()

    eval_mse = 0
    n_samples = 0
    for batch in tqdm(dataloader, desc="eval"):
        batch = tuple(t.to(device) for t in batch)
        b_input_t_ids, b_input_t_mask, b_input_d_ids, b_input_d_mask, b_labels = batch
        
        with torch.no_grad():        
            preds = model(b_input_t_ids, b_input_t_mask, b_input_d_ids, b_input_d_mask)
        
        b_labels = b_labels.detach().cpu().numpy().reshape(-1,1)
        preds = preds.detach().cpu().numpy().reshape(-1,1)

        #b_labels_rescaled = scaler.inverse_transform(b_labels)
        #preds_rescaled = scaler.inverse_transform(preds)
        
        #tmp_eval_mse = np.abs(preds_rescaled - b_labels_rescaled).sum()
        tmp_eval_mse = np.abs(preds - b_labels).sum()
        #import pdb;pdb.set_trace()
        eval_mse += tmp_eval_mse
        n_samples += b_input_d_ids.shape[0]

    return eval_mse/n_samples


bert_model = BertModel.from_pretrained("bert-base-uncased")
model = BERT_Regression(bert_model, args.hidden_size, args.n_highway,args.dropout)
model = model.to(device)

# train_dataloader, validation_dataloader, test_dataloader = get_all_dataloader(args.data_dir, args.processed_dir, args.batch_size)
train_dataloader, validation_dataloader, test_dataloader, scaler =  get_all_dataloader(args.processed_dir, args.datafile, args.dictfile, args.batch_size)

print ("    Number of training examples ", len(train_dataloader))
print ("    Number of dev examples ", len(validation_dataloader))
print ("    Number of test examples ", len(test_dataloader))

optimizer = AdamW(model.BERT_base.parameters(),lr = args.lr)
optimizer2 = AdamW([
            {'params': model.linear_model.parameters()},
            {'params': model.rhn.parameters()} 
            ],lr = args.lr2)
total_steps = len(train_dataloader) * args.epochs
if new_version:
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = int(0.1*total_steps),
                                            num_training_steps = total_steps)
else:
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                        warmup_steps = int(0.1*total_steps),
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
        b_input_t_ids, b_input_t_mask, b_input_d_ids, b_input_d_mask, b_labels = batch
                
        preds = model(b_input_t_ids, b_input_t_mask, b_input_d_ids, b_input_d_mask)
        
        loss = ((preds - b_labels)**2).sum()

        loss.backward()
        total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer2.step()
        optimizer.step()
        scheduler.step()
        model.zero_grad()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)            
    
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    val_mse = get_mse(model, validation_dataloader, scaler)
    if val_mse < best_val_mse:
        best_val_mse = val_mse
    test_mse = get_mse(model, test_dataloader, scaler)
    print(" Val mse {}, test mse {}".format(val_mse, test_mse))

print("")
print("Training complete!")
                                       


