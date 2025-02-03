#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys 
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/env/lib/python3.7/site-packages')
sys.path.append('/pl/active/banich/studies/Clearvale/jake_scripts/Amy_flywheel_scripts/')

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr
from scipy import stats
from scipy.spatial import distance
import math
import networkx as nx
import numpy as np
from igraph import *
from scipy.stats import zscore


# In[ ]:


eigcent = pd.read_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/dispersion_data/derosa_task_network_eigcent.csv')
disp = pd.read_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/dispersion_data/derosa_task_network_dispersion.csv')

z_data = pd.read_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/ClearMem_Z_Average.csv')
z_data = z_data[['SubID', 'z_ave', 'PSWQ_total', 'WBSI_total', 'RRS_total', 'RRS_depression', 'RRS_brooding', 'RRS_reflection']]
z_data = z_data.dropna()

eigcent = pd.read_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/dispersion_data/derosa_task_network_eigcent.csv')
disp = pd.read_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/dispersion_data/derosa_task_network_dispersion.csv')

z_data = pd.read_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/ClearMem_Z_Average.csv')
z_data = z_data[['SubID', 'z_ave', 'PSWQ_total', 'WBSI_total', 'RRS_total', 'RRS_depression', 'RRS_brooding', 'RRS_reflection']]
z_data = z_data.dropna()

z_data['b_z_ave'] = (zscore(z_data['PSWQ_total']) + zscore(z_data['WBSI_total']) + zscore(z_data['RRS_brooding']))/3
z_data['brd_z_ave'] = (zscore(z_data['PSWQ_total']) + zscore(z_data['WBSI_total']) + zscore(z_data['RRS_brooding']) + zscore(z_data['RRS_reflection']) + zscore(z_data['RRS_depression']))/5

sub_data = pd.merge(z_data, eigcent, on='SubID')
sub_data = pd.merge(sub_data, disp, on='SubID')

sub_data.to_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/sub_disp_eig_z_data.csv', index=False)


# In[ ]:


data = sub_data



# In[ ]:


import pandas as pd
from sklearn.preprocessing import power_transform

# Assuming sub_data is your DataFrame and filtered_data is already defined
# filtered_data = sub_data.filter(regex='disp')
# Filter columns based on regex
filtered_data = sub_data.filter(regex='disp')

# Apply Yeo-Johnson transformation
# Note: power_transform returns a numpy array, so we need to convert it back to a DataFrame
yeo_johnson_transformed_data = pd.DataFrame(power_transform(filtered_data, method='yeo-johnson'), columns=filtered_data.columns)



def get_disp_means(data):

    disp_z = data.filter(regex='SubID|disp|RRS|total|z_ave')
    
    # Compute mean across the specified columns for each subject
    disp_z['vn_mean'] = disp_z[['main_vn_dispersion_md', 'replace_vn_dispersion_md', 'suppress_vn_dispersion_md', 'clear_vn_dispersion_md']].mean(axis=1)
    # Compute mean across the specified columns for each subject
    disp_z['smn_mean'] = disp_z[['main_smn_dispersion_md', 'replace_smn_dispersion_md', 'suppress_smn_dispersion_md', 'clear_smn_dispersion_md']].mean(axis=1)
    # Compute mean across the specified columns for each subject
    disp_z['fpcn_mean'] = disp_z[['main_fpcn_dispersion_md', 'replace_fpcn_dispersion_md', 'suppress_fpcn_dispersion_md', 'clear_fpcn_dispersion_md']].mean(axis=1)
    # Compute mean across the specified columns for each subject
    disp_z['dmn_mean'] = disp_z[['main_dmn_dispersion_md', 'replace_dmn_dispersion_md', 'suppress_dmn_dispersion_md', 'clear_dmn_dispersion_md']].mean(axis=1)
    # Compute mean across the specified columns for each subject
    disp_z['overall_mean'] = disp_z[['main_vn_dispersion_md', 'replace_vn_dispersion_md', 'suppress_vn_dispersion_md', 'clear_vn_dispersion_md', 'main_fpcn_dispersion_md', 'replace_fpcn_dispersion_md', 'suppress_fpcn_dispersion_md', 'clear_fpcn_dispersion_md', 'main_dmn_dispersion_md', 'replace_dmn_dispersion_md', 'suppress_dmn_dispersion_md', 'clear_dmn_dispersion_md']].mean(axis=1)

    
     # Compute mean across the specified columns for each subject
    disp_z['main_mean'] = disp_z[['main_vn_dispersion_md', 'main_smn_dispersion_md', 'main_fpcn_dispersion_md', 'main_dmn_dispersion_md']].mean(axis=1)
    disp_z['replace_mean'] = disp_z[['replace_vn_dispersion_md', 'replace_smn_dispersion_md', 'replace_fpcn_dispersion_md', 'replace_dmn_dispersion_md']].mean(axis=1)
    disp_z['suppress_mean'] = disp_z[['suppress_vn_dispersion_md', 'suppress_smn_dispersion_md', 'suppress_fpcn_dispersion_md', 'suppress_dmn_dispersion_md']].mean(axis=1)
    disp_z['clear_mean'] = disp_z[['clear_vn_dispersion_md', 'clear_smn_dispersion_md', 'clear_fpcn_dispersion_md', 'clear_dmn_dispersion_md']].mean(axis=1)

    return disp_z

sub_disp = get_disp_means(sub_data)
sub_data_yeo = pd.concat([sub_data.filter(regex=f'^(?!.*(disp|eig)).*$'), yeo_johnson_transformed_data], axis=1)
sub_disp_yeo = get_disp_means(sub_data_yeo)


def run_regression(data, target, y_vars):

    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    
         # Joining list elements with ' + '
    joined_vars = ' + '.join(y_vars)
    # Turning it into a list with a single string element
    new_y_vars = [joined_vars]

    formula = f'{target} ~ {new_y_vars}'
    formula = formula.replace("[", "").replace("]", "").replace("'", "")

    # Fit the regression model using the formula
    model = smf.ols(formula=formula, data=data).fit()

    # Print the full regression output
    summary = model.summary()

    var = pd.DataFrame(summary.tables[0].data).iloc[0,1]

    table1 = pd.DataFrame(summary.tables[0].data).iloc[2:4, 3:].T.assign(var = var)
    table1.columns = ['fstat', 'pval', 'var']
    table1 = table1[['var', 'fstat', 'pval']]

    table2_cols = pd.DataFrame(summary.tables[1].data).loc[0].to_list() + ['var']
    table2_cols[0] = 'parameter'
    table2 = pd.DataFrame(summary.tables[1].data).iloc[1:].assign(var = var)
    table2.columns = table2_cols
    table2 = table2[['var', 'parameter', 'coef', 'std err', 't', 'P>|t|']]


    df = pd.merge(table1, table2, how='outer', left_on='var', right_on='parameter')

    # Combine 'var_x' and 'var_y' into a new column 'var'
    df['var'] = df['var_x'].fillna(df['var_y'])

    # Drop the original 'var_x' and 'var_y' columns
    df.drop(columns=['var_x', 'var_y'], inplace=True)

    # Reorder columns to place 'var' at the front if desired
    cols = ['var'] + [col for col in df.columns if col != 'var']
    df = df[cols]
    df = df.assign(formula = formula)
    
    return df


# In[261]:


def regression_function(data, variable_list):

    import itertools

    # Given list
    variables = variable_list
    targets = [['RRS_total'], ['PSWQ_total'], ['WBSI_total'], ['b_z_ave'], ['z_ave'], ['brd_z_ave']]

    # Create a list to hold all combinations
    all_combinations = []

    # Generate combinations for each length from 1 to the length of the list
    for r in range(1, len(variables) + 1):
        combinations = list(itertools.combinations(variables, r))
        all_combinations.extend(combinations)

    # Convert each tuple to a list
    list_combinations = [list(item) for item in all_combinations]

    regression_combinations = []
    for i in targets:
        for j in list_combinations:
            regression_combinations.append(run_regression(data, i, j))

    output_regressions = pd.concat(regression_combinations)

    # List of columns to convert
    columns_to_convert = ['fstat', 'pval', 'coef', 'std err', 't', 'P>|t|']

    # Convert each specified column to numeric, handling non-numeric values by converting them to NaN
    for column in columns_to_convert:
        output_regressions[column] = pd.to_numeric(output_regressions[column], errors='coerce')

    return output_regressions


# In[262]:


disp_mean_list = ['vn_mean', 'smn_mean', 'fpcn_mean', 'dmn_mean', 'overall_mean']


# In[263]:


disp_mean_reg = regression_function(sub_disp, disp_mean_list).query('pval < .05')

disp_mean_reg.to_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/operation_regressions/disp_mean_reg.csv', index=False)


# In[264]:


disp_mean_reg_yeo = regression_function(sub_disp_yeo, disp_mean_list).query('pval < .05')

disp_mean_reg.to_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/operation_regressions/disp_mean_reg_yeo.csv', index=False)


# In[265]:


disp_op_list = ['main_mean', 'replace_mean', 'suppress_mean', 'clear_mean']


# In[266]:


disp_op_reg = regression_function(sub_disp, disp_op_list).query('pval < .05')

disp_mean_reg.to_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/operation_regressions/disp_op_reg.csv', index=False)


# In[267]:


disp_op_reg_yeo = regression_function(sub_disp_yeo, disp_op_list).query('pval < .05')

disp_op_reg_yeo.to_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/operation_regressions/disp_op_reg_yeo.csv', index=False)


# In[ ]:


# List of items
disp_net_list = [
    'main_vn_dispersion_md', 'main_smn_dispersion_md',
    'main_fpcn_dispersion_md', 'main_dmn_dispersion_md',
    'replace_vn_dispersion_md', 'replace_smn_dispersion_md',
    'replace_fpcn_dispersion_md', 'replace_dmn_dispersion_md',
    'suppress_vn_dispersion_md', 'suppress_smn_dispersion_md',
    'suppress_fpcn_dispersion_md', 'suppress_dmn_dispersion_md',
    'clear_vn_dispersion_md', 'clear_smn_dispersion_md',
    'clear_fpcn_dispersion_md', 'clear_dmn_dispersion_md'
]

from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import itertools
import statsmodels.api as sm
import statsmodels.formula.api as smf

def regression_md_function(data, variable_list):

    import itertools

    # Given list
    variables = variable_list
    targets = [['RRS_total'], ['PSWQ_total'], ['WBSI_total'], ['b_z_ave'], ['z_ave'], ['brd_z_ave']]

    # Create a list to hold all combinations
    all_combinations = []

    # Generate combinations for each length from 1 to the length of the list
    for r in range(1, len(variables) + 1):
        combinations = list(itertools.combinations(variables, r))
        all_combinations.extend(combinations)

    # Convert each tuple to a list
    list_combinations = [list(item) for item in all_combinations]

    # Filter the list of lists to include only those with elements containing "DMN" or "FPCN"
    filtered_combinations = [sublist for sublist in list_combinations if any("dmn" in item or "fpcn" in item for item in sublist)]

    regression_combinations = []
    for i in targets:
        for j in filtered_combinations:
            regression_combinations.append(run_regression(data, i, j))

    output_regressions = pd.concat(regression_combinations)

    # List of columns to convert
    columns_to_convert = ['fstat', 'pval', 'coef', 'std err', 't', 'P>|t|']

    # Convert each specified column to numeric, handling non-numeric values by converting them to NaN
    for column in columns_to_convert:
        output_regressions[column] = pd.to_numeric(output_regressions[column], errors='coerce')

    return output_regressions


disp_net_reg = regression_md_function(sub_disp_yeo, disp_net_list).query('pval < .05')


disp_net_reg.to_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/operation_regressions/disp_net_reg.csv', index=False)
                    
                    


# In[ ]:


disp_net_reg = regression_md_function(sub_disp, disp_net_list).query('pval < .05')
disp_net_reg_yeo.to_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/operation_regressions/disp_net_reg_yeo.csv', index=False)
                


# In[ ]:





# In[ ]:




