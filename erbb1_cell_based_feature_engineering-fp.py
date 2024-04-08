## ---------------------------------------------------------------- ##
## --------- Feature Engineering with Cell Based Format ----------- ##
## ---------------------------------------------------------------- ##

from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
import numpy as np
from mordred import Calculator, descriptors

erbb1_df = pd.read_csv("erbb1_cellbased_neglog10_ic50.csv")

print("Dimensions for the initial SMILES dataset: ", erbb1_df.shape)

# Create a list for duplicate smiles
duplicates_smiles = erbb1_df[erbb1_df['canonical_smiles'].duplicated()]['canonical_smiles'].values
print("Number of duplicate SMILES: ", len(duplicates_smiles))

def morgan_fpts(data):
    Morgan_fpts = []
    for i in data:
        mol = Chem.MolFromSmiles(i)
        fpts =  AllChem.GetMorganFingerprintAsBitVect(mol,2,3072)
        mfpts = np.array(fpts)
        Morgan_fpts.append(mfpts)
    return np.array(Morgan_fpts)

Morgan_fpts = morgan_fpts(erbb1_df['canonical_smiles'])
print("Data dimensions before cleaning: ", str(Morgan_fpts.shape))

Morgan_fingerprints = pd.DataFrame(Morgan_fpts,columns=['Col_{}'.format(i) for i in range(Morgan_fpts.shape[1])])

counts = Morgan_fingerprints.nunique()
to_del = [i for i,v in enumerate(counts) if v == 1]
Morgan_fingerprints.drop(Morgan_fingerprints.columns[to_del], axis=1, inplace=True)
Morgan_fingerprints.drop_duplicates(inplace=True)
Morgan_fingerprints = Morgan_fingerprints.T.drop_duplicates().T
print("Data dimensions after cleaning: ", str(Morgan_fingerprints.shape))

output_file = 'cell-based_fingerprints.csv'  # Create the output textfile
with open(output_file, 'w') as file:  # Open the output text file to write in it
    file.write(Morgan_fingerprints.to_csv(index=False))
