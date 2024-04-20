from rdkit import Chem
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



class ToxicityDataset(Dataset):

    def __init__(self, smiles, herg, vocab, randomize=False, seq_len=256):
        self.smiles = smiles

        herg = (herg/100)
        print(min(herg), max(herg))
        herg = herg
        self.herg = herg
        self.vocab = vocab
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
        t = self.herg[item]
        if self.randomize:
            sm = self.randomize_smiles(sm)
        else:
            sm = Chem.CanonSmiles(sm)   
        content = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm]
        X = [self.vocab.sos_index] + content + [self.vocab.eos_index]
        padding = [self.vocab.pad_index]*(self.seq_len - len(X))
        X.extend(padding)
        return (torch.tensor(X).long(), torch.tensor(t).float())