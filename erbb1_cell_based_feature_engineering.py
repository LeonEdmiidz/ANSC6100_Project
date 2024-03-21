## ---------------------------------------------------------------- ##
## --------- Feature Engineering with Cell Based Format ----------- ##
## ---------------------------------------------------------------- ##

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import plotly.graph_objects as go

df = pd.read_csv("erbb1_cellbased_neglog10_ic50.csv")

def RDkit_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles] 
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()
    
    Mol_descriptors =[]
    for mol in mols:
        # add hydrogens to molecules
        mol=Chem.AddHs(mol)
        # Calculate all 200 descriptors for each molecule
        descriptors = calc.CalcDescriptors(mol)
        Mol_descriptors.append(descriptors)
    return Mol_descriptors,desc_names 

Mol_descriptors,desc_names = RDkit_descriptors(df['canonical_smiles'])

df_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)

df = pd.DataFrame(df_with_200_descriptors)
import re
from sklearn.feature_selection import VarianceThreshold


#-------Check for single value columns ----------

#Function to identify and remove single value columns in data.
def find_single_value_columns(dataframe):
    #Loop over columns and return those with only 1 unique value
    return [col for col in dataframe if dataframe[col].nunique() == 1]

#Update  user
single_value_columns = find_single_value_columns(df)
print("Single-value columns found and removed:")
for idx, column in enumerate(single_value_columns, 1):
    print(f"{column}")  #Print col name to user thats removed

#Identify and remove SVC's    
single_value_columns = find_single_value_columns(df)
df.drop(columns=single_value_columns, inplace=True)

#------- Remove columns with variance <1 ----------

selector = VarianceThreshold(threshold=1)  

df = df.select_dtypes(include=[np.number])
transformed_data = selector.fit_transform(df)

cols = df.columns[selector.get_support(indices=True)]

df = pd.DataFrame(transformed_data, columns=cols)