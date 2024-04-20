import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class DeepToxRegressor(nn.Module):
    def __init__(self, vocab_len, embed_dim=64, seq_len=128, smiles_proj_dim=64,dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_len, embed_dim)
        self.lin1 = nn.Linear(embed_dim, 256)
        self.lin2 = nn.Linear(seq_len, 128)
        self.SmilesProjLayer = nn.Linear(256*128, smiles_proj_dim)
        self.CombinedLayer1 = nn.Linear(smiles_proj_dim, 32)
        self.CombinedLayer2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, smiles_idx):
        #print(smiles_idx.size())
        sm_emb = self.emb(smiles_idx)
        #print(sm_emb.size())
        # BxSXK
        sm_proj_emb = self.lin1(sm_emb)
        sm_proj_emb = F.relu(sm_proj_emb)
        # BXSX8
        #print(sm_proj_emb.size())
        sm_proj_emb= sm_proj_emb.transpose(1,2)
        # BX8XS
        #print(sm_proj_emb.size())
        B = sm_emb.size()[0]
        sm_proj_emb = self.lin2(sm_proj_emb)
        sm_proj_emb = F.relu(sm_proj_emb)
        # BX8X32
        #print(sm_proj_emb.size())

        x = sm_proj_emb.view(B, 256*128)
        #print(x.size())
     #   v, _ = torch.max(sm_emb, 1)
     #   x = torch.mean(sm_emb,1)
        #print(x.size())
        #print(v.size())
       # res = torch.concat([x,v], dim=1)
        #print(res.size())
       # x = self.SmilesProjLayer(res)
      #  x = self.dropout(x)
      #  y = self.dropout(y)
        
        #x = torch.cat([x,y], dim=1)
        x = self.dropout(x)
        x = self.SmilesProjLayer(x)
        x = F.relu(x)
        x =self.CombinedLayer1(x)
        x = F.relu(x)
        x =self.CombinedLayer2(x)
        
        return x