#importing necessary libraries
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler



#loading in the data from the output of data_preprocessing.py
data = pd.read_csv("erbb1_singleprotein_neglog10_ic50.csv")
print(data.head())

#creating new features based on the canononical SMILES from data

def mol_descriptors(smiles):
    molecules = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()

    molecular_descriptors = []
    for mol in molecules:
        #add hydrogen to the molecules
        mol = Chem.AddHs(mol)
        #Calculate the molecular descriptors for each molecule
        descriptors = calc.CalcDescriptors(mol)
        molecular_descriptors.append(descriptors)
    return molecular_descriptors, desc_names

#calling the mol_descriptors function
molecular_descriptors, desc_names = mol_descriptors(data["canonical_smiles"])

#creating a dataframe for the moleculare descriptors
data_descriptors = pd.DataFrame(molecular_descriptors, columns=desc_names)

print(data_descriptors.shape)

#saving raw molecular descriptors to a csv file
data_descriptors.to_csv("raw_descriptors_single_protein.csv", index = False)

##Eliminate single value columns

#getting the number of unique values in each column
num_unique_col = data_descriptors.nunique()

#getting a record of single-value columns
col_to_del = [i for i,v in enumerate(num_unique_col) if v == 1 ]

#drop single value columns
data_descriptors.drop(data_descriptors.columns[col_to_del], axis=1, inplace=True)

var_list = data_descriptors.apply(np.var, axis=0)

##dropping columns with variance less than 1

#define variance threshold
var_threshold = VarianceThreshold(1)

var_threshold.fit(data_descriptors)

feature_mask = var_threshold.get_support()

selected_columns = data_descriptors.columns[feature_mask]

data_descriptors = data_descriptors[selected_columns]

print(data_descriptors.shape)


print(data_descriptors.head())


#add the response (ic50)

data_descriptors["standardized_ic50"] = data["-log(M)"]

print(data_descriptors.head())

nan_mask = data_descriptors.isna().any(axis = 1)
rows_with_nan = data_descriptors[nan_mask]
print(rows_with_nan)

data_descriptors = data_descriptors.dropna()

scaler = MinMaxScaler()

#Don't scale the last column
features = data_descriptors.columns[:-1]
data_descriptors[features] = scaler.fit_transform(data_descriptors[features])

print(data_descriptors.head())


data_descriptors.to_csv("df_for_model_building.csv", index=False)
