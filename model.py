import torch
import torch.nn as nn
from transformers import BertModel

class JargonFakeNewsModel(nn.Module):
    def __init__(self):
        super(JargonFakeNewsModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.gru = nn.GRU(768, 128, num_layers=1, bidirectional=True, batch_first=True)
        self.attention_W = nn.Linear(256, 128)
        self.attention_v = nn.Linear(128, 1)
        self.fc = nn.Linear(256, 1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids, mask):
        out = self.bert(ids, attention_mask=mask)
        # We use the sequence output for the GRU
        gru_output, _ = self.gru(out.last_hidden_state)
        M = torch.tanh(self.attention_W(gru_output))
        a = torch.softmax(self.attention_v(M), dim=1)
        r = torch.sum(a * gru_output, dim=1)
        return self.sigmoid(self.fc(r))
