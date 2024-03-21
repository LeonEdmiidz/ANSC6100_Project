
## ------------------------------------------------------ ##
## --------- Dataframe for Cell-Based Format ----------- ##
## ------------------------------------------------------ ##

#importing necessary libraries
import numpy as np
import pandas as pd
import plotly.graph_objects as go
#from chembl_webresource_client.new_client import new_client

# #searching and selecting  as the drug target
# target_query = new_client.target.search("Erbb1")
# targets = pd.DataFrame(target_query)
# print(targets.iloc[1,:])

# #select the first option as selected_query
# selected_query = targets.target_chembl_id[0]
# print(selected_query)

#selecting the desired data set
#activity = new_client.activity
#erbb1_ic50 = activity.filter(target_chembl_id = selected_query).filter(standard_type = "IC50")
#erbb1_df = pd.DataFrame(erbb1_ic50)
#print(erbb1_df)

#importing the data set
erbb1_df = pd.read_csv("raw_data_erbb1_ic50.csv")
print(erbb1_df.head())

#removing any missing values in the "standard value" column
erbb1_df_naremove = erbb1_df[erbb1_df.standard_value.notna()]
print(erbb1_df_naremove.shape)

#removing missing cananonical smiles
erbb1_df_naremove_smiles = erbb1_df_naremove[erbb1_df_naremove.canonical_smiles.notna()]
print(erbb1_df_naremove_smiles.shape)

#Getting a list of columns from the erbb1_df
print(erbb1_df_naremove_smiles.columns)

#looking at the data_validity_comments
print(erbb1_df_naremove_smiles['data_validity_comment'].value_counts())

#selecting desired columns
selected_columns = ['canonical_smiles', 'molecule_chembl_id','bao_label', 'standard_units', 'standard_value','data_validity_comment']
erbb1_df_select_columns = erbb1_df_naremove_smiles[selected_columns]
print(erbb1_df_select_columns.head())

#checking out the out of range values
erbb1_df_err = erbb1_df_select_columns[erbb1_df_select_columns['data_validity_comment'] == 'Outside typical range' ]
print(erbb1_df_err.loc[:,'standard_value'].head())

#checking the max and min for the standard value of erbb1_df_err
erbb1_df_err['standard_value'] = pd.to_numeric(erbb1_df_err['standard_value'])
print(erbb1_df_err['standard_value'].max())
print(erbb1_df_err['standard_value'].min())

#filtering for data_validity_comment = None
erbb1_df_err_rm = erbb1_df_select_columns[erbb1_df_select_columns['data_validity_comment'].isnull()]
print(erbb1_df_err_rm.shape)

#looking at range of standard_values
erbb1_df_err_rm['standard_value'] = pd.to_numeric(erbb1_df_err_rm['standard_value'])
print(erbb1_df_err_rm['standard_value'].max())
print(erbb1_df_err_rm['standard_value'].min())

#looking at the bao labels (experimental design) and looking at the number of observations for each bao label
print(erbb1_df_err_rm['bao_label'].unique())
print(erbb1_df_err_rm['bao_label'].value_counts())

#creating a function to take the -log of the molar value of nM
import math

#function to convert the nM of a compound to the -log10(m)
def logm(nm):
    m  = nm/1000000000
    m = -math.log10(m)
    return m

#Selecting `cell based format' boa labels
erbb1_cbf_df = erbb1_df_err_rm[erbb1_df_err_rm['bao_label'] == 'cell-based format']
print(erbb1_cbf_df.shape)

#Removing duplicate compounds (molecular_chembl_id)
erbb1_cbf_nonduplicates_df = erbb1_cbf_df.drop_duplicates(subset=['molecule_chembl_id'])
print(erbb1_cbf_nonduplicates_df.shape)

#Removing duplicate compounds (canonical SMILES)
erbb1_cbf_nonduplicates_df = erbb1_cbf_nonduplicates_df.drop_duplicates(subset=['canonical_smiles'])
print(erbb1_cbf_nonduplicates_df.shape)

#creating a new column called '-log(M)' which contains the -log(M) of the 'standard_value' column
erbb1_cbf_nonduplicates_df['-log(M)'] = erbb1_cbf_nonduplicates_df['standard_value'].apply(logm)

print(erbb1_cbf_nonduplicates_df.loc[:,'-log(M)'].head())

#check to make sure we don't have NA's in our columns
print(erbb1_cbf_nonduplicates_df.isna().sum())

#subsetting for canonical_smiles and -log(M)
final_df = erbb1_cbf_nonduplicates_df[['canonical_smiles', '-log(M)']]
print(final_df)

final_df.to_csv("erbb1_cellbased_neglog10_ic50.csv", index = False)

bao_labels = erbb1_cbf_nonduplicates_df['bao_label'].unique()

## PLOTTING DISTRIBUTION OF LOG TRANSFORMED VALUES

fig = go.Figure()

for bao_label in bao_labels:
    df_filtered = erbb1_cbf_nonduplicates_df[erbb1_cbf_nonduplicates_df['bao_label'] == bao_label]
    fig.add_trace(go.Box(
        y=df_filtered['-log(M)'],
        name=bao_label,
        boxpoints='all',  
        jitter=0.5,  
        whiskerwidth=0.2,
        marker_size=2,
        line_width=1)
    )

fig.update_layout(
    title='Distribution of -log transformed standard values',
    yaxis=dict(
        title='-log(M)',
        autorange=True,
        showgrid=True,
        zeroline=True,
        dtick=5,
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',
        zerolinewidth=2,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
    showlegend=True
)

fig.show()

