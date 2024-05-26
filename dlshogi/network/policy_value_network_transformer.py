import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=96):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        return x + self.pos_encoding[:, :x.size(1)]


class PolicyValueNetwork(nn.Module):
    def __init__(self, ntoken=96, d_model=256, nhead=8, dim_feedforward=256, num_layers=8, dropout=0.1):
        super(PolicyValueNetwork, self).__init__()
        self.encoder = nn.EmbeddingBag(2892, d_model, mode="sum", padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, ntoken)
        transformer_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)
        self.policy_conv = nn.Conv1d(d_model, d_model // 8, 1, bias=False)
        self.policy_norm = nn.BatchNorm1d(ntoken * d_model // 8)
        self.policy_fc = nn.Linear(ntoken * d_model // 8, 1496)
        self.value_conv = nn.Conv1d(d_model, d_model // 8, 1, bias=False)
        self.value_norm1 = nn.BatchNorm1d(ntoken * d_model // 8)
        self.value_fc1 = nn.Linear(ntoken * d_model // 8, 256, bias=False)
        self.value_norm2 = nn.BatchNorm1d(256)
        self.value_fc2 = nn.Linear(256, 1)
        self.ntoken = ntoken
        self.d_model = d_model
        

    def forward(self, src):
        x = self.encoder(src.type(torch.int64).view(-1, 35))
        x = x.view(-1, self.ntoken, self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        policy = self.policy_conv(x)
        policy = F.relu(self.policy_norm(policy.flatten(1)))
        policy = self.policy_fc(policy)
        value = self.value_conv(x)
        value = F.relu(self.value_norm1(value.flatten(1)))
        value = F.relu(self.value_norm2(self.value_fc1(value)))
        value = self.value_fc2(value)
        return policy, value