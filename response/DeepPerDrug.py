# Now lets try with torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class PersonalizedDrugResModel(nn.Module):
  
    def __init__(self, vocab_len, embed_dim=64, seq_len = 64, numRNA_features=17737,mRNA_proj_dim=1024, smiles_proj_dim=64,dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_len, embed_dim)
        self.lin1 = nn.Linear(seq_len, 64)
        self.lin2 = nn.Linear(64, 16)
        self.lin3 = nn.Linear(16*embed_dim, 256*4)
        self.dropout = nn.Dropout(dropout)
        self.dropoutSm = nn.Dropout(0.1)
        self.layernormSmiles = nn.LayerNorm(1024)
        self.layernormRNA = nn.LayerNorm(1024)
        self.mRNAProjLayer1 = nn.Linear(numRNA_features, mRNA_proj_dim)
        self.mRNAProjLayer2 = nn.Linear(mRNA_proj_dim, 256*4)
        
        self.FinalLayer1 = nn.Linear(256*4+256*4, 128)
        self.FinalLayer2 = nn.Linear(128, 64)
        self.FinalLayer3 = nn.Linear(64, 32)
        self.FinalLayer4 = nn.Linear(32, 1)

    def forward(self, smiles_idx, mRNA_feat):
        smiles_idx = self.dropoutSm(smiles_idx.float())
        sm_emb = self.emb(smiles_idx.long())
        # BxSXK
        #sm_proj_emb = self.lin1(sm_emb)
        #sm_proj_emb = F.relu(sm_proj_emb)
        # BXSX8
        sm_proj_emb= sm_emb.transpose(1,2)
        # BX8XS
        B = sm_emb.size()[0]
        sm_proj_emb = self.lin1(sm_proj_emb)
        sm_proj_emb = F.relu(sm_proj_emb)
        sm_proj_emb = self.lin2(sm_proj_emb)
        sm_proj_emb = F.relu(sm_proj_emb)
        # BX8X32
        x_smiles = sm_proj_emb.view(B, sm_proj_emb.numel()//B)
        x_smiles = self.lin3(x_smiles)
        x_smiles = F.relu(x_smiles)       
 
    
        mRNA_feat = mRNA_feat
        mRNA_feat = self.dropout(mRNA_feat)       
        x = self.mRNAProjLayer1(mRNA_feat)
        x = F.relu(x)
        x_mRNA =self.mRNAProjLayer2(x)
        x_mRNA = F.relu(x_mRNA)
        #print(x_smiles.size(), x_mRNA.size())
        x_mRNA = self.layernormRNA(x_mRNA)
        x_smiles = self.layernormSmiles(x_smiles)
        x = torch.cat([x_smiles, x_mRNA], dim=1)
        #x = x_mRNA
        #print(x.size())
        x = self.FinalLayer1(x)

        x = F.relu(x)
        x = self.FinalLayer2(x)
        x = F.relu(x)
        x = self.FinalLayer3(x)
        x = F.relu(x)
        x = self.FinalLayer4(x)
      #  x =self.CombinedLayer1(x)
      #  x =self.CombinedLayer2(x)
        
        return x