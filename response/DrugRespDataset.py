from rdkit import Chem
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



class DrugRespDataset(Dataset):

    def __init__(self, smiles,gene_expr, tgt, vocab, randomize=True, seq_len=256):
        self.smiles = smiles
        X_norm = gene_expr-np.mean(gene_expr, axis=0)
        X_norm = X_norm/np.std(X_norm, axis=0)
        # X_range = np.max(X_norm)-np.min(X_norm)
        self.gene_expr = X_norm
        self.vocab = vocab
        self.tgt = tgt
        self.seq_len = seq_len
        self.randomize = randomize
        print(len(self.smiles), type(self.smiles))
    def __len__(self):
        return len(self.smiles)

    def randomize_smiles(self, smiles):
        """Perform a randomization of a SMILES string
        must be RDKit sanitizable"""
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None # Invalid SMILES
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        return Chem.MolToSmiles(nm, canonical=False, isomericSmiles=False)    
    
    def __getitem__(self, item):
        sm = self.smiles[item]
        gene_expr = self.gene_expr[item]
        t = self.tgt[item]
        if self.randomize:
       # sm = Chem.CanonSmiles(sm)
            sm = self.randomize_smiles(sm)
        else:
            sm = Chem.CanonSmiles(sm)   
        #sm = self.transform(sm) # List
        content = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm]
        X = [self.vocab.sos_index] + content + [self.vocab.eos_index]
        padding = [self.vocab.pad_index]*(self.seq_len - len(X))
        X.extend(padding)
 #       print(torch.tensor(X))
        return (torch.tensor(X).long(), torch.tensor(gene_expr).float(), torch.tensor(t))