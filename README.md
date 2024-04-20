# MediSeeker
 Code to build a pipeline for drug discovery for personalized cancer treatment. This repo consists of the
 following directories:
 - toxicity: Code for training a toxicity predictor (regressor) on the herG dataset
 - response: Code for training a predictor to estimate the IC50 value for a cell line, drug combination
 - ilm: Code for training an infilling generator model to "fill in the blanks" given a smiles representation. Modified from code in  https://github.com/chrisdonahue/ilm/tree/master. We modify the code to mask smiles representation of drugs and train on a nanoGPT model architecture
 - st: Utilities for tokenization of smiles strings. Modified from code in https://github.com/DSPsleeporg/smiles-transformer

 - nanoGPT: We only use the model definition (modified) from (https://github.com/karpathy/nanoGPT) for infilling.



## References
- [SMILES Transformer: Pre-trained Molecular Fingerprint for Low Data Drug Discovery](https://arxiv.org/abs/1911.04738) by Shion Honda et al.
- [_Enabling language models to fill in the blanks_](https://arxiv.org/abs/2005.05339) (Donahue et al. 2020)  
- tdcommons for all datasets https://github.com/mims-harvard/TDC

## Getting started
- Install pytorch 2.1, tdc, numpy, pandas, sklearn and tqdm
- Train the models for toxicity prediction and drug response prediction using the provided notebooks
- (Optional) Train the infilling model for filling blanks in SMILES strings (GPU recommended)
- Alternately, use the pretrained infilling model provided.
- Once all models are trained run the notebook MediSeekerDemo.ipynb
