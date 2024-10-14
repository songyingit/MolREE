import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from rdkit import Chem
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def molformer_embs_gen(smiles_list, model, tokenizer, batch_size=1024):
    """
    Generate MolFormer embeddings from SMILES of REE binding molecules.

    Parameters:
    smiles_list (list): List of SMILES strings.
    model (AutoModel): Pretrained MolFormer model.
    tokenizer (AutoTokenizer): Tokenizer corresponding to the MolFormer model.
    batch_size (int): Batch size for processing SMILES.

    Returns:
    np.ndarray: Array of embeddings.
    """
    embeddings = []
    for i in tqdm(range(0, len(smiles_list), batch_size)):
        batch_smiles = smiles_list[i:i+batch_size]
        tokenized_smiles = tokenizer(batch_smiles, padding=True, return_tensors="pt")
        with torch.no_grad():
            output = model(**tokenized_smiles)
        batch_embeddings = output.pooler_output
        embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(embeddings)

if __name__ == "__main__":

    # Load dataset
    ds_process_file = pd.ExcelFile('REE_Dataset.xlsx')
    df = pd.read_excel(ds_process_file)

    # Get SMILES list and validate SMILES via RDKit
    smiles_list = df['Canonical SMILES'].to_list()
    rdkit_smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smile)) for smile in smiles_list]

    # Load pre-trained MolFormer model and tokenizer
    model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

    # Generate embeddings for a long list of SMILES
    batch_size = 1024
    embeddings_array = molformer_embs_gen(rdkit_smiles_list, model, tokenizer, batch_size)
    print(embeddings_array.shape)

    # Save embeddings to file
    np.save('molformer_embeddings.npy', embeddings_array)
