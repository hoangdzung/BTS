import torch
import torch.nn as nn 
from rhn import HighwayBlock

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)

class BERT_Regression(nn.Module):
    """The main model."""
    def __init__(self, bert_model, hidden_size, n_highway=4, dropout=0.5):
        super().__init__()
        self.n_highway = n_highway
        self.BERT_base =  bert_model
        self.rhn = HighwayBlock(768)
        self.dropout= nn.Dropout(dropout)
        self.linear_model = nn.Sequential(
            #nn.Linear(768*2, hidden_size),
            nn.Dropout(dropout),
            #nn.ReLU(),
            #nn.Linear(768, 1),
            nn.Linear(768, 1),
            #nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, token_ids, masks, token_ids2, masks2):
        hidden_bert1, out_bert1 = self.BERT_base(token_ids, attention_mask=masks)
        hidden_bert2, out_bert2 = self.BERT_base(token_ids2, attention_mask=masks2)
        out_bert1 = hidden_bert1.mean(1)
        out_bert2 = hidden_bert2.mean(1)
        # out_bert = torch.cat([out_bert1, out_bert2], dim=1)
        out_bert = (out_bert1+out_bert2)/2
        out_bert = self.dropout(out_bert)
        for _ in range(self.n_highway):
            out_bert = self.rhn(out_bert)
        out = self.linear_model(out_bert)
        #out2 = torch.sigmoid(out_bert.mean(1,keepdim=True))
        return out
        return (out+out2)/2
