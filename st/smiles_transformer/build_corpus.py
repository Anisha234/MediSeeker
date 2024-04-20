import argparse
import pandas as pd
from tqdm import tqdm
from build_vocab import WordVocab
from rdkit import Chem

from utils import split
from tdc.generation import MolGen
def main():
    parser = argparse.ArgumentParser(description='Build a corpus file')
   # parser.add_argument('--in_path', '-i', type=str, default='data/chembl24_bert_train.csv', help='input file')
    parser.add_argument('--out_path', '-o', type=str, default='data/chembl_corpus.txt', help='output file')
    args = parser.parse_args()

#    smiles = pd.read_csv(args.in_path)['first'].values
    data = MolGen(name = 'ChEMBL_V29')
    split2 = data.get_split()
    df = split2['train'][0:1000000]
    df['smiles'] = df['smiles'].apply(lambda x: Chem.CanonSmiles(x))
    # Get length of the smiles string in a separate column
    df['Len'] = df['smiles'].apply(lambda x: len(list(x)))

# Drop rows where the length of the smiles string is longer than 120
    filtered_df = df[df['Len'] <= 250]
    df['smiles'] = df['smiles'].apply(lambda x: split(x)+'\n')
   # smiles = list(split2['train']['smiles'])
   # smiles = smiles
   # print(len(smiles))
    with open(args.out_path, 'a') as f:
        f.write("".join(df['smiles']))
    #    for sm in tqdm(smiles):
    #        sm = Chem.CanonSmiles(sm)
    #        print((sm), type(sm))
    #        print(split((sm)))
    #        f.write(split(sm)+'\n')
    print('Built a corpus file!')

    with open(args.out_path, "r", encoding='utf-8') as f:
        vocab = WordVocab(f, max_size=None, min_freq=500)
    
    print(vocab.freqs)
    print(vocab.stoi)
    print(len(vocab.itos))
    pd.Series(vocab.stoi.keys()).to_csv('data/chembl_vocab.csv',index=False, lineterminator="\n", header=False)
    vocab.save_vocab('data/chembl_vocab.pkl')
if __name__=='__main__':
    main()



