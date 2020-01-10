import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

# Highway block for each recurrent 'tick'
class HighwayBlock(nn.Module):
    def __init__(self, hidden_size):
        super(HighwayBlock, self).__init__()

        # hidden weight matrices
        self.R_H = nn.Linear(hidden_size, hidden_size)
        self.R_T = nn.Linear(hidden_size, hidden_size)
        #self.R_C = nn.Linear(hidden_size, hidden_size)


    def forward(self, prev_hidden):
        hl = self.R_H(prev_hidden)
        #tl = self.R_C(prev_hidden)
        cl = torch.sigmoid(self.R_T(prev_hidden))

        # Core recurrence operation
        _hidden = hl * (1-cl) + (prev_hidden * cl)
        return F.relu(_hidden)
