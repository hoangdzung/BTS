import torch
import torch.nn as nn 

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)

class BERT_Regression(nn.Module):
    """The main model."""
    def __init__(self, bert_model, hidden_size, dropout=0.5):
        super().__init__()
        self.BERT_base =  bert_model
        self.linear_model = nn.Sequential(
            # nn.Linear(768, hidden_size),
            nn.Dropout(dropout),
            # nn.PReLU(),
            # nn.Linear(hidden_size, 1)
            nn.Linear(768, 1)
        )
        self.apply(weights_init)

    def forward(self, token_ids, masks, token_ids2, masks2):
        hidden_bert1, _ = self.BERT_base(token_ids, attention_mask=masks)
        hidden_bert2, _ = self.BERT_base(token_ids2, attention_mask=masks2)
        out_bert1 = hidden_bert1.mean(1)
        out_bert2 = hidden_bert2.mean(1)
        #out_bert = torch.mean([out_bert1, out_bert2], dim=1)
        out_bert = out_bert1+out_bert2
        out = self.linear_model(out_bert)
        
        return out
