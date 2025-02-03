#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys 
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/env/lib/python3.7/site-packages')
sys.path.append('/pl/active/banich/studies/Clearvale/jake_scripts/Amy_flywheel_scripts/')

import numpy as np
import pandas as pd
import os


# In[4]:


import sys
# adding Folder_2 to the system path
sys.path.insert(0, '/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/leiden/analysis/')
from clearmem_my_functions import *

import sys
sys.path.append('/home/jade6100/.local/lib/python3.7/site-packages')
import scikit_posthocs as sp
import itertools

import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

import warnings
import sys 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os
import numpy as np
import pandas as pd
import scipy.io
from scipy import stats
from sklearn.manifold import MDS
import scipy.spatial.distance as sp_distance
import matplotlib.pyplot as plt
import seaborn as sns 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

#import statsmodels.api as sm
#from statsmodels.formula.api import ols
#from scipy import stats
#import scikit_posthocs as sp

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold

def getF(data, var, group):
    data[var] = data[var].astype('float')
    model = ols(var + '~ C('+group+')', data=data).fit()
    anova_table = np.array(sm.stats.anova_lm(model, type=2)[['F', 'PR(>F)']])[0]
    return anova_table.round(4)

def getposthoc(data, var, group):
    data[var] = data[var].astype('float')
    model = ols(var + '~ C('+group+')', data=data).fit()
    post_hoc = sp.posthoc_ttest(data, val_col=var, group_col=group,
                                p_adjust='fdr_bh').sort_index().sort_index(axis = 1)
    post_hoc.columns = list(range(1, len(uni(data[group]))+1))
    return post_hoc

def vsim(data, parcel=None):
    
    if parcel is None:
        parcel_row = pd.DataFrame(np.flip(np.array(data.T))).T
        
    else:
        parcel_row = pd.DataFrame(np.flip(np.array(data.iloc[[parcel]].T))).T
    a = np.array(parcel_row.T.iloc[:, 0: 1]).squeeze()
    n = int(np.sqrt(len(a)*2))+1
    mask = np.tri(n,dtype=bool, k=-1) # or np.arange(n)[:,None] > np.arange(n)
    out = np.zeros((n,n))
    out[mask] = a
    out = out + out.T - np.diag(np.diag(out))
    out[np.diag_indices_from(out)]
    sim_mat = pd.DataFrame(np.flip(out))
  

    return(sim_mat)
    
def grad_centroid_vsim(data, grad): 
    new_row_cluster = data[data['grad'] == grad]
    new_row_cluster = new_row_cluster.drop(['grad'], axis=1)
    clust_centroid = pd.DataFrame(new_row_cluster.mean(), columns = ['Mean']).T
    
    sim_mat = vsim(clust_centroid)
    
    return(sim_mat)


def threshold_proportional(W,p,copy=True):

    if p>1 or p<0:
        raise BCTParamError('Threshold must be in range [0,1]')
    if copy: W=W.copy()
    n=len(W)						# number of nodes
    np.fill_diagonal(W, 0)			# clear diagonal

    if np.all(W==W.T):				# if symmetric matrix
        W[np.tril_indices(n)]=0		# ensure symmetry is preserved
        ud=2						# halve number of removed links
    else:
        ud=1

    ind=np.where(W)					# find all links

    I=np.argsort(W[ind])[::-1]		# sort indices by magnitude

    en=int(round((n*n-n)*p/ud))		# number of links to be preserved

    W[(ind[0][I][en:],ind[1][I][en:])]=0	# apply threshold
    #W[np.ix_(ind[0][I][en:], ind[1][I][en:])]=0

    if ud==2:						# if symmetric matrix
        W[:,:]=W+W.T						# reconstruct symmetry

    return W


import plotly.graph_objs as go
from plotly.subplots import make_subplots
import itertools

def plot_3Dfigure(newX, colors, title='', net=None, filepath=None):
    fig = plt.figure()
    
    if net is not None:
        data = go.Scatter3d(x=newX[:,0], y=newX[:,1], z=newX[:,2], 
                            mode='markers',
                            marker=dict(size=5,
                                        #color=newX[:,1],
                                        color=colors,
                                        opacity=0.7,
                                        colorscale=colors)
                           )
    else:
        data = go.Scatter3d(x=newX[:,0], y=newX[:,1], z=newX[:,2], 
                            mode='markers',
                            marker=dict(size=5,
                                        color=newX[:,1],
                                        opacity=0.7,
                                        colorscale='jet')
                           )
    
    layout = go.Layout(title_text=title,title_x=0.5,title_y=0.8,title_font_size=12)
    fig = go.Figure(data=[data], layout=layout)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(scene = dict(
                    xaxis = dict(title= '', ticks= '', showticklabels= False,),
                    yaxis = dict(title= '', ticks= '', showticklabels= False,),
                    zaxis = dict(title= '', ticks= '', showticklabels= False,),
                    ))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    #fig.show()
    
    if filepath is not None:
        fig.write_html(filepath+'3d_grads.html')
        

from pylab import *

def get_color_maps(cmap, n, nrep):
    from PIL import Image, ImageColor

    cmap = cm.get_cmap(cmap, n)    # PiYG
    converted_list = []
    for i in range(cmap.N):
        rgba = cmap(i)
        # rgb2hex accepts rgb or rgba
        hexc = matplotlib.colors.rgb2hex(rgba)
        codes = ImageColor.getcolor(str(hexc), "RGB")

        c1 = list(itertools.repeat(int(codes[0])/255, nrep))
        c2 = list(itertools.repeat(int(codes[1])/255, nrep))
        c3 = list(itertools.repeat(int(codes[2])/255, nrep))
        
        converted = pd.concat([pd.DataFrame(c1), pd.DataFrame(c2), pd.DataFrame(c3)], axis = 1)
        converted.columns = ['r', 'g', 'b']
        converted_list.append(converted)
        
    final = pd.concat(converted_list).reset_index(drop=True)
    final['index_new'] = list(range(0,final.shape[0]))
    
    return final

def show_grads(data):
    import nibabel as nib
    import nilearn.plotting as plotting
    import numpy as np
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    import pandas as pd
    import hcp_utils as hcp

    def listToDict(lstA, lstB):
        zipped = zip(lstA, lstB)
        op = dict(zipped)
        return op

    def uni(list1):
        unique_list = []
        # traverse for all elements
        for x in list1:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)

        return(unique_list)

    glasser = pd.read_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/leiden/analysis/glasser_regions/spearman_subtype_glasser_regions.csv').drop('regionID', axis =1).rename({'Unnamed: 0':'regionID'}, axis = 1)
    glasser['regionID'] = glasser['regionID'] + 1
    
    mesh_path = '/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/leiden/hcp-utils/hcp_utils/data/'
    mesh_sub = hcp.load_surfaces(example_filename=mesh_path+'S1200.R.pial_MSMAll.32k_fs_LR.surf.gii')
    
    regs = data 
    regs['regionID'] = regs['index'] + 1
    regs = pd.merge(regs, glasser, on = 'regionID')
    regs = regs[['regionID', 'Subtype_x', 'grad', 'regionName', 'regionLongName', 'Lobe', 'cortex', 'r', 'g', 'b']]
  
    maps_df = pd.DataFrame(hcp.mmp.rgba).T
    maps_df = maps_df.reset_index().rename({'index':'regionID'}, axis = 1)
    maps_df['regionID'] = maps_df['regionID'] +1
    maps_df = pd.DataFrame(hcp.mmp.rgba).T
    maps_df = maps_df.reset_index().rename({'index':'regionID'}, axis = 1)
    maps_df['regionID'] = maps_df['regionID'] +1

    new_maps_df = pd.merge(regs, maps_df, on = "regionID").sort_values('regionID')
    new_maps_df[0] = new_maps_df['r']
    new_maps_df[1] = new_maps_df['g']
    new_maps_df[2] = new_maps_df['b']
    new_maps_df = new_maps_df[[0, 1,2,3]]

    first = pd.DataFrame(np.array([0, 0, 0, 0])).T
    more = pd.concat([pd.DataFrame(np.zeros(20)), pd.DataFrame(np.zeros(20)), pd.DataFrame(np.zeros(20)), pd.DataFrame(np.zeros(20))], axis= 1)
    more.columns = [0,1,2,3]

    new_maps_df = pd.concat([first, new_maps_df, more])

    map_l = []
    out_l = []
    for i in range(0, len(new_maps_df)):
        map_l.append(i)
        out_l.append(np.array(new_maps_df.iloc[i:i+1])[0])

    hcp.mmp.rgba = listToDict(map_l, out_l)

    return hcp.view_parcellation(mesh_sub.inflated, hcp.mmp)

def get_colors(cmap, n):
        from PIL import Image, ImageColor

        cmap = cm.get_cmap(cmap, n)    # PiYG
        converted_list = []
        for i in range(cmap.N):
            rgba = cmap(i)
            # rgb2hex accepts rgb or rgba
            hexc = matplotlib.colors.rgb2hex(rgba)
            converted_list.append(hexc)

        return converted_list
    
    
def show_code(function):  
    import inspect
    lines = inspect.getsource(function)
    print(lines)
    
jets = get_colors('jet_r', 20)


net_cols = ['#FCFF0D', '#21DFB4', '#4E00A2', '#F00087']
ops_cols = ['#F0180A', '#F08B0A', '#6DAE45', '#0A5AF0']



def reduce_memory_usage(df, verbose=False):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


# In[5]:



from glob import glob
subj_ids = '/pl/active/banich/studies/wmem/fmri/operation_rsa/subj/*'
subj_ids_sorted = sorted(glob(subj_ids, recursive = True))

for i in range(0,len(subj_ids_sorted)):
    subj_ids_sorted[i] = subj_ids_sorted[i].replace('/pl/active/banich/studies/wmem/fmri/operation_rsa/subj/subj_operation_sub-', '').replace('/', '').replace('.mat', '')
    #subj_ids_sorted[i] = 'sub' + subj_ids_sorted[i] + "-" + str(i+1)


# In[6]:


def create_func_mat(data):
    
    from sklearn.preprocessing import MinMaxScaler
    
    X = np.array(data)
    
    X_Z = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, dtype=np.float64, ddof=1, keepdims=True)
    D, rho = spearmanr(np.array(X_Z), axis=1)

    perc = np.array([np.percentile(x, 90) for x in D])

    for i in range(D.shape[0]):
        D[i, D[i,:] < perc[i]] = 0    
    
    D[D < 0] = 0
    #D = 1 - pairwise_distances(D, metric = 'cosine')
    #scaler = MinMaxScaler(feature_range=(.5,1))

    #D = scaler.fit_transform(D)
    
    D = pd.DataFrame(D)
    # D = aff
    
    return(D)


# In[7]:


group = pd.read_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/leiden/group_sm_vector_spearman/Output/Results/group_sm_vector_spearman_Full_Subtypes.csv')

gmat_main = pd.DataFrame(create_func_mat(group.iloc[:, 5:].iloc[:, 0:10332]))
gmat_replace = pd.DataFrame(create_func_mat(group.iloc[:, 5:].iloc[:, 10332:int(10332*2)]))
gmat_suppress = pd.DataFrame(create_func_mat(group.iloc[:, 5:].iloc[:, int(10332*2):int(10332*3)]))
gmat_clear = pd.DataFrame(create_func_mat(group.iloc[:, 5:].iloc[:, int(10332*3):int(10332*4)]))

group_subs = group[['Unnamed: 0', 'Subtype']].rename({'Unnamed: 0':'index'}, axis=1)


# In[8]:


glob_path = '/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/leiden/subj_sm_ind/*'
from glob import glob

sub_idx = glob(glob_path, recursive = True)

for i in range(0,len(sub_idx)):
    sub_idx[i] = sub_idx[i].replace('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/leiden/subj_sm_ind/', '').replace('/', '')
    
sub_idx = list(filter(lambda k: 'sub' in k, sub_idx))

sub_idx = list(filter(lambda k: '.csv' not in k, sub_idx))

#remove sub019 because no accuracy or evdidence classifiers
sub_idx = sorted([x for x in sub_idx if 'sub019_sm_vector' not in x])
#sub_idx = sub_idx[0:5]


# In[9]:


def get_new_mats(data, parcel_index, gfm):
    
    sfm = pd.DataFrame(create_func_mat(data))
    sfm.columns = parcel_index
    sfm.index = parcel_index
    smf_cols = list(pd.DataFrame(sfm).columns)

    not_in_list = [x for x in list(range(360)) if x not in smf_cols]
    
    sfm_copy = sfm.copy()
    for i in not_in_list:
        sfm_copy[i] = gfm[i]
        
    sfm_copy = sfm_copy.T

    for i in not_in_list:
        sfm_copy[i] = gfm[i]

    sfm_copy = pd.DataFrame(sfm_copy.sort_index().sort_index(axis=1))


    return(sfm_copy)


# In[10]:


def get_mats(data_path, group_subs):
    
    df1 = pd.read_csv(data_path).drop('Subtype', axis =1)

    main_cols = list(map(str,range(0, 10332)))
    replace_cols = list(map(str,range(10332, 10332*2)))
    suppress_cols = list(map(str,range(10332*2, 10332*3)))
    clear_cols = list(map(str,range(10332*3, 10332*4)))
    
    lh = list(df1[df1['node'].str.contains("LH")]['node'].str.replace(r'\D', '').astype(int) -1)
    rh = list(df1[df1['node'].str.contains("RH")]['node'].str.replace(r'\D', '').astype(int) + 179)

    parcel_index = lh + rh
    df1.index = parcel_index
    
    not_in_list = [x for x in list(range(360)) if x not in parcel_index]
    
    df1_cols = list(df1.iloc[:, 4:].columns)
    
    df1['index'] = df1.index
    df1 = pd.merge(df1, group_subs, on = 'index')
    df1.index = parcel_index
    
    
    if len(parcel_index) < 360:
         
        ts1_subs = df1[['index', 'Subtype']]
        add_subs = group_subs.T[not_in_list].T
        copy_frame = df1
        ts1_added_subs = pd.concat([ts1_subs, add_subs])
        ts1_added_subs = pd.concat([copy_frame.drop(['index', 'Subtype'], axis=1), ts1_added_subs], axis=1)

        df1 = ts1_added_subs
        

    return df1, parcel_index, not_in_list
    


# In[11]:


class subj_outputs:

    def __init__(self, path):
        self.path = path
        self.df1, self.parcel_index, self.not_in_list = get_mats(self.path, group_subs)


# In[12]:


subj_path = []
for i in sub_idx:
    subj_path.append('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/leiden/subj_sm_ind/'+i+'/Output/Results/'+i+'_Full_Subtypes.csv')

from timeit import default_timer as timer
start = timer()
sub_class_list = *map(subj_outputs, subj_path),
end = timer()
print(end - start)


# In[13]:


for i,j in zip(range(len(sub_class_list)), sub_idx):
        print('----------------------- processed:'+j)
        os.system(f'mkdir -p /pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/subj/{j}')
        sub_class_list[i].name = j


# In[14]:


plot_path = '/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/subj/'
for i in range(len(sub_class_list)):
    sub_class_list[i].plot_path = plot_path + sub_class_list[i].name+'/'+sub_class_list[i].name+'_'


# In[33]:


#sub_class_list[0].df1[main_cols]

def sub_trial_cor(sub_num):
    
    global sub_class_list
    
    def get_parc_cors(sub_class_list, sub_num, op_name):
        
        if op_name == 'main':
            op_list = list(map(str,range(0, 10332)))
    
        if op_name == 'replace':
             op_list = list(map(str,range(10332, 10332*2)))

        if op_name == 'suppress':
             op_list = list(map(str,range(10332*2, 10332*3)))

        if op_name == 'clear':
             op_list = list(map(str,range(10332*3, 10332*4)))

        
        columns_to_select = [col for col in op_list if col in sub_class_list[sub_num].df1.columns]
        selected_df = sub_class_list[sub_num].df1[columns_to_select]
     
        parc_cors =[]
        #for i in sub_class_list[sub_num].parcel_index:
        for i in range(360):
            parc_cors.append(selected_df.iloc[i].mean())
             #parc_cors.append(selected_df.query('parcel in @i').mean())
    
        parc_cors_df = (pd.DataFrame(parc_cors, columns=[f'{op_name}_cors'])
                        .assign(sub = sub_class_list[sub_num].name[3:6]).iloc[:, [1,0]]
                       )
        
        return parc_cors_df

    sub_op_cors=[]
    for i in ['main', 'replace', 'suppress', 'clear']:
        sub_op_cors.append(get_parc_cors(sub_class_list, sub_num, i))
        
    global group
    
    networks = group[['Key', 'Subtype']]
    networks['parcel'] = networks['Key'].astype(int)
    networks = networks[['parcel', 'Subtype']]
    
    mapping = {1: 'VN', 2: 'SMN', 3: 'FPCN', 4: 'DMN'}
    networks['Subtype'] = networks['Subtype'].map(mapping)

    sub_op_cors_df = pd.concat(sub_op_cors, axis=1).iloc[:, [0,1,3,5,7]].dropna()
    sub_op_cors_df['parcel'] = sub_class_list[sub_num].parcel_index 
    sub_op_cors_df['parcel'] = sub_op_cors_df['parcel'] + 1
    
    sub_op_cors_df = pd.merge(sub_op_cors_df, networks, on='parcel')
    sub_op_cors_df = sub_op_cors_df[['sub', 'parcel', 'Subtype', 'main_cors', 'replace_cors', 'suppress_cors', 'clear_cors']]
        
        
    #sub_name =  sub_op_cors_df['sub'].unique()[0]
    #sub_cors_net = sub_op_cors_df.groupby('Subtype').mean().reset_index().drop('parcel', axis=1).melt(id_vars='Subtype')
    
    #sub_cors_net['net_op_cor'] = sub_cors_net['variable'] + '_' + sub_cors_net['Subtype']
    
    #sub_cors_net = (sub_cors_net[['net_op_cor', 'value']].reset_index()
    #                 .pivot(index='index', columns='net_op_cor', values='value')
    #                 .reset_index(drop=True)
    #                ).melt().dropna().T
    #sub_cors_net.columns = sub_cors_net.iloc[0].to_list()
    #sub_cors_net = pd.DataFrame(sub_cors_net.iloc[1]).T.reset_index(drop=True)
    
    # Calculate the variance by network instead of the mean
    sub_cors_net = sub_op_cors_df.groupby('Subtype').var().reset_index().drop('parcel', axis=1).melt(id_vars='Subtype')

    # Create the 'net_op_cor' column by combining 'variable' and 'Subtype'
    sub_cors_net['net_op_cor'] = sub_cors_net['variable'] + '_' + sub_cors_net['Subtype']

    # Reshape the DataFrame to have 'net_op_cor' as columns and 'value' as values
    sub_cors_net = (sub_cors_net[['net_op_cor', 'value']].reset_index()
                    .pivot(index='index', columns='net_op_cor', values='value')
                    .reset_index(drop=True)
                   ).melt().dropna().T
    sub_cors_net.columns = sub_cors_net.iloc[0].to_list()
    sub_cors_net = pd.DataFrame(sub_cors_net.iloc[1]).T.reset_index(drop=True)
    sub_cors_net['sub'] = sub_class_list[sub_num].name[3:6]
    # good 
    op_cors = pd.DataFrame(sub_op_cors_df.mean()).T.iloc[:, 2:]
    op_cors['sub'] = sub_class_list[sub_num].name[3:6]

    #all_cors = op_cors
    all_cors = pd.merge(op_cors, sub_cors_net, on='sub')
    all_cors= all_cors[['sub'] + [col for col in all_cors.columns if col != 'sub']]

    return all_cors


# In[34]:


all_sub_cors_df = pd.concat([sub_trial_cor(i) for i in range(len(sub_class_list))])


# In[ ]:


all_sub_cors_df.to_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/bootstrapped_regressions/sub_rsa_mat_cors.csv', index=False)


# In[2]:


#!jupyter nbconvert --to script sub_rsa_mat_cors.ipynb


# In[ ]:




