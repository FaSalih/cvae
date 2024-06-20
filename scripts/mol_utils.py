# this can be loaded because it's added to the path
# using __init__.py (check it.)
# run as python  -m scripts.mol_utils.py

from chemvae.mol_utils import load_smiles_and_data_df
import numpy as np

if __name__ == "__main__":
    # smiles, reg_dat, logit_dat = load_smiles_and_data_df("models/zinc/250k_rndm_zinc_drugs_clean_3.csv", 120,
    #  ['logP', 'qed', 'SAS'], ['NRingsGT6', 'PAINS'])

    smiles, reg_dat, _ = load_smiles_and_data_df(
        "models/zinc/250k_rndm_zinc_drugs_clean_3.csv",
        120,  # type: ignore
        ["logP", "qed", "SAS"],
    )
    print(smiles[:5])
    print(reg_dat[:5, :]) if isinstance(reg_dat, np.ndarray) else print(
        "No regression data."
    )
    # print(logit_dat[:5, :])
