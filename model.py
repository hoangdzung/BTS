import torch.nn as nn 

class BERT_Regression(nn.Module):
    """The main model."""
    def __init__(self, bert_model, hidden_size, dropout=0.5):
        super().__init__()
        self.BERT_base =  bert_model
        self.linear_model = nn.Sequential(
            # nn.Linear(768, hidden_size),
            # nn.Dropout(dropout),
            # nn.PReLU(),
            # nn.Linear(hidden_size, 1)
            nn.Linear(768, 1)
        )
    
    def forward(self, token_ids, masks):
        _, out_bert = self.BERT_base(token_ids, attention_mask=masks)
        out = self.linear_model(out_bert)
        
        return out