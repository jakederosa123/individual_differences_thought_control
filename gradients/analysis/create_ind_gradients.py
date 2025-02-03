#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
    
jets = get_colors('jet', 20)


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


# In[ ]:



from glob import glob
subj_ids = '/pl/active/banich/studies/wmem/fmri/operation_rsa/subj/*'
subj_ids_sorted = sorted(glob(subj_ids, recursive = True))

for i in range(0,len(subj_ids_sorted)):
    subj_ids_sorted[i] = subj_ids_sorted[i].replace('/pl/active/banich/studies/wmem/fmri/operation_rsa/subj/subj_operation_sub-', '').replace('/', '').replace('.mat', '')
    #subj_ids_sorted[i] = 'sub' + subj_ids_sorted[i] + "-" + str(i+1)


# In[ ]:


evidence_glob_path = '/pl/active/banich/studies/wmem/fmri/mvpa/utaustin/data/clearmem_v1_*/mvpa_operation/parse_sh10_4/table_target_evidence_operation_highres001_brain_grey_3mm_bin0.2_bold_mcf_brain_hpass_dt_shift10tr_norest_fixpen50.csv'
accuracy_glob_path = '/pl/active/banich/studies/wmem/fmri/mvpa/utaustin/data/clearmem_v1_*/mvpa_operation/parse_sh10_4/table_target_accuracy_operation_highres001_brain_grey_3mm_bin0.2_bold_mcf_brain_hpass_dt_shift10tr_norest_fixpen50.csv'
from glob import glob

evidence_sub_idx = sorted(glob(evidence_glob_path, recursive = True))
accuracy_sub_idx = sorted(glob(accuracy_glob_path, recursive = True))

for i in range(0,len(evidence_sub_idx)):
    evidence_sub_idx[i] = evidence_sub_idx[i].replace('/pl/active/banich/studies/wmem/fmri/mvpa/utaustin/data/clearmem_v1_', '').replace('sub', '')
    evidence_sub_idx[i] = evidence_sub_idx[i].replace('/mvpa_operation/parse_sh10_4/table_target_evidence_operation_highres001_brain_grey_3mm_bin0.2_bold_mcf_brain_hpass_dt_shift10tr_norest_fixpen50.csv', '')

for i in range(0,len(accuracy_sub_idx )):
    accuracy_sub_idx[i] = accuracy_sub_idx[i].replace('/pl/active/banich/studies/wmem/fmri/mvpa/utaustin/data/clearmem_v1_', '').replace('sub', '')
    accuracy_sub_idx[i] = accuracy_sub_idx[i].replace('/mvpa_operation/parse_sh10_4/table_target_accuracy_operation_highres001_brain_grey_3mm_bin0.2_bold_mcf_brain_hpass_dt_shift10tr_norest_fixpen50.csv', '')

evidence_sub_ids_df = pd.DataFrame(evidence_sub_idx).rename({0:'ID'}, axis=1)
accruacy_sub_ids_df = pd.DataFrame(accuracy_sub_idx).rename({0:'ID'}, axis=1)

ev_acc_ids_df = pd.merge(evidence_sub_ids_df, accruacy_sub_ids_df, on ='ID')

sub_ids_df = pd.DataFrame(subj_ids_sorted).rename({0:'ID'}, axis=1)
sub_ids_df['matched_id'] = list(range(1,56,1))
 
matched_ids_df = pd.merge(ev_acc_ids_df , sub_ids_df, on ="ID")


# In[ ]:


add_zero_list =[]
for i in list(matched_ids_df['matched_id']):
    
    if len(str(i)) == 1:
        x = "00" + str(i)
    else: 
        x = "0" + str(i)
    
    add_zero_list.append(x)
    
matched_ids_df['matched_id'] = add_zero_list

accuracy_list = []
evidence_list = []
for i,j in zip(sorted(glob(accuracy_glob_path, recursive = True)),
               sorted(glob(evidence_glob_path, recursive = True))):
               accuracy_list.append(pd.read_csv(i)[['subject_id', 't1_maintain', 't2_repCat', 't3_target', 't4_global', 'total_accuracy']])
               evidence_list.append(pd.read_csv(j))


# In[ ]:


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


# In[ ]:


group = pd.read_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/leiden/group_sm_vector_spearman/Output/Results/group_sm_vector_spearman_Full_Subtypes.csv')

gmat_main = pd.DataFrame(create_func_mat(group.iloc[:, 5:].iloc[:, 0:10332]))
gmat_replace = pd.DataFrame(create_func_mat(group.iloc[:, 5:].iloc[:, 10332:int(10332*2)]))
gmat_suppress = pd.DataFrame(create_func_mat(group.iloc[:, 5:].iloc[:, int(10332*2):int(10332*3)]))
gmat_clear = pd.DataFrame(create_func_mat(group.iloc[:, 5:].iloc[:, int(10332*3):int(10332*4)]))

group_subs = group[['Unnamed: 0', 'Subtype']].rename({'Unnamed: 0':'index'}, axis=1)


# In[ ]:


glob_path = '/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/leiden/subj_sm_ind/*'
from glob import glob

sub_idx = glob(glob_path, recursive = True)

for i in range(0,len(sub_idx)):
    sub_idx[i] = sub_idx[i].replace('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/leiden/subj_sm_ind/', '').replace('/', '')
    
sub_idx = list(filter(lambda k: 'sub' in k, sub_idx))

sub_idx = list(filter(lambda k: '.csv' not in k, sub_idx))

#remove sub019 because no accuracy or evdidence classifiers
#sub_idx = sorted([x for x in sub_idx if 'sub019_sm_vector' not in x])
sub_idx = sorted([x for x in sub_idx if 'sub019_sm_vector' in x])

#sub_idx = sub_idx[0:3]


# In[ ]:


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


# In[ ]:


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
    
    main_cols_pull = [x for x in df1_cols if x in main_cols]
    replace_cols_pull = [x for x in df1_cols if x in replace_cols]
    suppress_cols_pull = [x for x in df1_cols if x in suppress_cols]
    clear_cols_pull = [x for x in df1_cols if x in clear_cols]
    
    main = df1[main_cols_pull]
    replace = df1[replace_cols_pull]
    suppress = df1[suppress_cols_pull]
    clear = df1[clear_cols_pull]
    
    main.name = 'main'
    replace.name = 'replace'
    suppress.name = 'suppress'
    clear.name = 'clear'
    
    if len(parcel_index) < 360:
        main_mat = get_new_mats(main, parcel_index, gmat_main)
        replace_mat = get_new_mats(replace, parcel_index, gmat_replace)
        suppress_mat = get_new_mats(suppress, parcel_index, gmat_suppress)
        clear_mat = get_new_mats(clear, parcel_index, gmat_clear)
         
        ts1_subs = df1[['index', 'Subtype']]
        add_subs = group_subs.T[not_in_list].T
        copy_frame = df1
        ts1_added_subs = pd.concat([ts1_subs, add_subs])
        ts1_added_subs = pd.concat([copy_frame.drop(['index', 'Subtype'], axis=1), ts1_added_subs], axis=1)

        df1 = ts1_added_subs
        
    else: 
        main_mat = create_func_mat(main)
        replace_mat = create_func_mat(replace)
        suppress_mat = create_func_mat(suppress)
        clear_mat = create_func_mat(clear)
                                           
     
    main_mat = reduce_memory_usage(main_mat)
    replace_mat = reduce_memory_usage(replace_mat)
    suppress_mat = reduce_memory_usage(suppress_mat)
    clear_mat = reduce_memory_usage(clear_mat)
        

    colors = df1.Subtype.map({1:net_cols[0], 2:net_cols[1], 3:net_cols[2], 4:net_cols[3]})
    

    return df1, parcel_index, not_in_list, colors, main, replace, suppress, clear, main_mat, replace_mat,suppress_mat, clear_mat
    


# In[ ]:



class subj_outputs:

    def __init__(self, path):
        self.path = path
        self.df1, self.parcel_index, self.not_in_list, self.colors, self.main, self.replace, self.suppress, self.clear, self.main_mat, self.replace_mat, self.suppress_mat, self.clear_mat = get_mats(self.path, group_subs)


# In[ ]:


subj_path = []
for i in sub_idx:
    subj_path.append('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/leiden/subj_sm_ind/'+i+'/Output/Results/'+i+'_Full_Subtypes.csv')


from timeit import default_timer as timer
start = timer()
sub_class_list = *map(subj_outputs, subj_path),
end = timer()
print(end - start)


# In[ ]:


old_reads = '''subj_path = []
for i in sub_idx:
    subj_path.append('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/leiden/subj_sm_ind/'+i+'/Output/Results/'+i+'_Full_Subtypes.csv')

subj_path = subj_path[0:10]

from timeit import default_timer as timer
start = timer()

sub_class_list = []
for i in subj_path:

    class subj_outputs:

        subj_path = i

        def __init__(self, subj_path):

            #print(self.subj_path)

            self.df1, self.parcel_index, self.not_in_list, self.colors, self.main, self.replace, self.suppress, self.clear, self.main_mat, self.replace_mat, self.suppress_mat, self.clear_mat = get_mats(self.subj_path, group_subs)

    sub_class_list.append(subj_outputs(i))


end = timer()
print(end - start)
'''


# In[ ]:


for i,j in zip(range(len(sub_class_list)), sub_idx):
        print('----------------------- processed:'+j)
        os.system(f'mkdir -p /pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/subj/{j}')
        sub_class_list[i].name = j


# In[ ]:


plot_path = '/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/subj/'
for i in range(len(sub_class_list)):
    sub_class_list[i].plot_path = plot_path + sub_class_list[i].name+'/'+sub_class_list[i].name+'_'


# In[ ]:


old_all_mats ='''all_mats = []
for i in range(len(sub_class_list)):
    main_mat_list = sub_class_list[i].main_mat
    replace_mat_list =  sub_class_list[i].replace_mat
    suppress_mat_list = sub_class_list[i].suppress_mat
    clear_mat_list = sub_class_list[i].clear_mat 
    
    all_mats.append(main_mat_list)
    all_mats.append(replace_mat_list)
    all_mats.append(suppress_mat_list)
    all_mats.append(clear_mat_list)
'''


# In[ ]:


def get_grads(mat_list):
    
    from brainspace.gradient import GradientMaps
    from brainspace.plotting import plot_hemispheres
    from brainspace.utils.parcellation import map_to_labels
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from brainspace.null_models import SurrogateMaps
    import seaborn as sns
    
    gj = GradientMaps(approach='dm',
                      kernel='normalized_angle',
                      alignment='joint',
                      random_state=1)
    

    global group_grads
    gj_out = gj.fit(mat_list)
    

    for i in range(1):
        sns.set_context("paper", font_scale = 2)
        
        fig, ax = plt.subplots(1, figsize=(5, 4))
        ax.scatter(range(gj_out.lambdas_[i].size), gj_out.lambdas_[i])
        ax.set_xlabel('Component')
        ax.set_ylabel('Eigenvalue')
        fig.tight_layout()
        fig.savefig('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/grp/eigen_components.png')
  
        fig, ax = plt.subplots(1, figsize=(5, 4))
        variance = gj_out.lambdas_[0]
        ax.scatter(range(1,11), variance/variance.sum())
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, end+.05, 0.05))
        ax.set_xlabel('Component')
        ax.set_ylabel('variance %')
        variance/variance.sum()
        fig.tight_layout()
        fig.savefig('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/grp/variance_explained.png')
    
    return gj_out


# In[ ]:


gmats = [gmat_main, gmat_replace, gmat_suppress, gmat_clear]
group_grads = get_grads(gmats)


# In[ ]:


for i in range(4):
    group_grads.aligned_[i][:,1] = group_grads.aligned_[i][:,1]*-1


# In[ ]:


main_grp_grads = pd.DataFrame(group_grads.aligned_[0]).iloc[:,0:3]
main_grp_grads.columns =['g1', 'g2', 'g3']
main_grp_grads.to_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/grp/main_grads.csv')


# In[ ]:


replace_grp_grads = pd.DataFrame(group_grads.aligned_[1]).iloc[:,0:3]
replace_grp_grads.columns =['g1', 'g2', 'g3']
replace_grp_grads.to_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/grp/replace_grads.csv')


# In[ ]:


suppress_grp_grads = pd.DataFrame(group_grads.aligned_[2]).iloc[:,0:3]
suppress_grp_grads.columns =['g1', 'g2', 'g3']
suppress_grp_grads.to_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/grp/suppress_grads.csv')


# In[ ]:


clear_grp_grads = pd.DataFrame(group_grads.aligned_[3]).iloc[:,0:3]
clear_grp_grads.columns =['g1', 'g2', 'g3']
clear_grp_grads.to_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/grp/clear_grads.csv')


# In[ ]:


#sub_class_list = sub_class_list_copy


# In[ ]:


from brainspace.gradient import GradientMaps
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from brainspace.null_models import SurrogateMaps
import seaborn as sns


def align_subs(data):
    
    global group_grads
    
    main_mat_list = data.main_mat
    replace_mat_list =  data.replace_mat
    suppress_mat_list = data.suppress_mat
    clear_mat_list = data.clear_mat 
    
    gj = GradientMaps(approach='dm', kernel='normalized_angle', alignment='procrustes', random_state=1)
    gj_out_main = gj.fit(np.array(main_mat_list), reference = group_grads.aligned_[0])
    print('finished main alignment for: ' +  data.name)
    
    gj = GradientMaps(approach='dm', kernel='normalized_angle', alignment='procrustes', random_state=1)
    gj_out_replace = gj.fit(np.array(replace_mat_list), reference =group_grads.aligned_[1])
    print('finished replace alignment for: ' +  data.name)
    
    gj = GradientMaps(approach='dm', kernel='normalized_angle', alignment='procrustes', random_state=1)
    gj_out_suppress = gj.fit(np.array(suppress_mat_list), reference = group_grads.aligned_[2])
    print('finished suppress alignment for: ' +  data.name)
    
    gj = GradientMaps(approach='dm', kernel='normalized_angle', alignment='procrustes', random_state=1)
    gj_out_clear = gj.fit(np.array(clear_mat_list), reference = group_grads.aligned_[3])
    print('finished clear alignment for: ' +  data.name)
        
    data.main_aligned = gj_out_main.aligned_
    data.replace_aligned = gj_out_replace.aligned_
    data.suppress_aligned = gj_out_suppress.aligned_
    data.clear_aligned = gj_out_clear.aligned_
    
    return data


# In[ ]:


from timeit import default_timer as timer
start = timer()
sub_class_list = *map(align_subs, sub_class_list),
end = timer()
print(end - start)


# In[ ]:


def process_grads(sub_class_list, op):
    
    if op == 'main':
        df =sub_class_list.main_aligned
        
    elif op == 'replace':
        df =sub_class_list.replace_aligned 
        
    elif op == 'suppress':
        df =sub_class_list.suppress_aligned
        
    elif op == 'clear':
        df =sub_class_list.clear_aligned
        
    op_mat = sub_class_list.df1
    
    filepath = sub_class_list.plot_path+'_'+op+'_'
    
    rc_g1=df[:,0]
    rc_g2=df[:,1]
    rc_g3=df[:,2]
    rc_g4=df[:,3]
    rc_g5=df[:,4]
    
    Y = np.stack((rc_g1, rc_g2, rc_g3, rc_g4, rc_g5)).T

    fig = plt.figure(figsize=(8.5,5.5))
  
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    colors = sub_class_list.colors

    ax1.scatter(Y[:, 0], Y[:, 1], c=Y[:, 1], cmap='jet_r')
    ax2.scatter(Y[:, 0], Y[:, 1], c=colors)

    fig.tight_layout()
    fig.savefig(f'{filepath}_grad_axes.png')

    plt.clf()
  
    #get_anim(Y)
    plot_3Dfigure(Y, colors, net=True, filepath=sub_class_list.plot_path+'network_')
    plot_3Dfigure(Y, colors, filepath=sub_class_list.plot_path)
    
    grads = pd.DataFrame(Y)
    #grads.index = parcel_index
    grads.columns = ['g1', 'g2', 'g3', 'g4', 'g5']
    df_grads = pd.concat([op_mat, grads], axis = 1)
    sub_grads = df_grads[['Subtype', 'g1', 'g2', 'g3', 'g4', 'g5']].reset_index()
    #sub_grads.index = parcel_index

   
    N = int(op_mat.shape[0])
    
    if N == 360:
        grad_nums = []
        for i in range(1,21):
            grad_nums.append(list(itertools.repeat(i, int(360/20))))
            #grad_nums = list(itertools.chain.from_iterable(grad_nums))
        grad_nums = list(pd.DataFrame(np.array(grad_nums)).melt().sort_values('value')['value'])
            
    else:
        rounded = round(N*0.05)
        num_bins = trunc(N/rounded)
        subtract = num_bins*rounded
        num_in_added_bin = op_mat.shape[0] - subtract
        bin_list = []
        for i in list(range(1, (int(num_bins)) + 1)):
            for j in range(rounded):
                bin_list.append(i)

        grad_nums = bin_list + [int(num_bins + 1)]*int(num_in_added_bin)

    g1_new_order = sub_grads[['index', 'Subtype', 'g1']].sort_values(by = 'g1')
    g1_new_order['grad'] = grad_nums
    g1_new_order['index_new'] = list(range(0,N))
    
    g2_new_order = sub_grads[['index', 'Subtype', 'g2']].sort_values(by = 'g2')
    g2_new_order['grad'] = grad_nums
    g2_new_order['index_new'] =  list(range(0,N))
    
    g3_new_order = sub_grads[['index', 'Subtype', 'g3']].sort_values(by = 'g3')
    g3_new_order['grad'] = grad_nums
    g3_new_order['index_new'] =  list(range(0,N))
    
    g4_new_order = sub_grads[['index', 'Subtype', 'g4']].sort_values(by = 'g4')
    g4_new_order['grad'] = grad_nums
    g4_new_order['index_new'] =  list(range(0,N))
    
    g5_new_order = sub_grads[['index', 'Subtype', 'g5']].sort_values(by = 'g5')
    g5_new_order['grad'] = grad_nums
    g5_new_order['index_new'] =  list(range(0,N))
    
    if op == 'main':
         sub_class_list.main_grads_processed = g1_new_order, g2_new_order, g3_new_order, g4_new_order, g5_new_order
    
    elif op == 'replace':
        sub_class_list.replace_grads_processed = g1_new_order, g2_new_order, g3_new_order, g4_new_order, g5_new_order
        
    elif op == 'suppress':
        sub_class_list.suppress_grads_processed = g1_new_order, g2_new_order, g3_new_order, g4_new_order, g5_new_order
        
    elif op == 'clear':
        sub_class_list.clear_grads_processed = g1_new_order, g2_new_order, g3_new_order, g4_new_order, g5_new_order
        
        
    return sub_class_list


# In[ ]:


from timeit import default_timer as timer
from functools import partial

for j in ['main', 'replace', 'suppress', 'clear']:
    sub_class_list = *map(partial(process_grads, op=j), sub_class_list),


# In[ ]:


old_process_grads = '''def process_grads(grad_fit, op_mat, num, colors, filepath):
    
    rc_g1=grad_fit.gradients_[num][:,0]
    rc_g2=grad_fit.gradients_[num][:,1]
    rc_g3=grad_fit.gradients_[num][:,2]
    rc_g4=grad_fit.gradients_[num][:,3]
    rc_g5=grad_fit.gradients_[num][:,4]
    
    Y = np.stack((rc_g1, rc_g2, rc_g3, rc_g4, rc_g5)).T

    fig = plt.figure(figsize=(8.5,5.5))
  
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.scatter(Y[:, 0], Y[:, 1], c=Y[:, 1], cmap='jet_r')
    ax2.scatter(Y[:, 0], Y[:, 1], c=colors)

    fig.tight_layout()
    fig.savefig(f'{filepath}_grad_axes.png')

    plt.clf()
  
    #get_anim(Y)
    plot_3Dfigure(Y, colors, net=True, filepath=None)
    plot_3Dfigure(Y, colors, filepath=None)
    
    grads = pd.DataFrame(Y)
    #grads.index = parcel_index
    grads.columns = ['g1', 'g2', 'g3', 'g4', 'g5']
    df_grads = pd.concat([op_mat, grads], axis = 1)
    sub_grads = df_grads[['Subtype', 'g1', 'g2', 'g3', 'g4', 'g5']].reset_index()
    #sub_grads.index = parcel_index

   
    N = int(op_mat.shape[0])
    
    if N == 360:
        grad_nums = []
        for i in range(1,21):
            grad_nums.append(list(itertools.repeat(i, int(360/20))))
            #grad_nums = list(itertools.chain.from_iterable(grad_nums))
        grad_nums = list(pd.DataFrame(np.array(grad_nums)).melt().sort_values('value')['value'])
            
    else:
        rounded = round(N*0.05)
        num_bins = trunc(N/rounded)
        subtract = num_bins*rounded
        num_in_added_bin = op_mat.shape[0] - subtract
        bin_list = []
        for i in list(range(1, (int(num_bins)) + 1)):
            for j in range(rounded):
                bin_list.append(i)

        grad_nums = bin_list + [int(num_bins + 1)]*int(num_in_added_bin)

    g1_new_order = sub_grads[['index', 'Subtype', 'g1']].sort_values(by = 'g1')
    g1_new_order['grad'] = grad_nums
    #g1_new_order['grad'] = list(range(0,N))
    g1_new_order['index_new'] = list(range(0,N))
    
    g2_new_order = sub_grads[['index', 'Subtype', 'g2']].sort_values(by = 'g2')
    g2_new_order['grad'] = grad_nums
    #g2_new_order['grad'] = list(range(0,N))
    g2_new_order['index_new'] =  list(range(0,N))
    
    g3_new_order = sub_grads[['index', 'Subtype', 'g3']].sort_values(by = 'g3')
    g3_new_order['grad'] = grad_nums
    #g3_new_order['grad'] = list(range(0,N))
    g3_new_order['index_new'] =  list(range(0,N))
    
    g4_new_order = sub_grads[['index', 'Subtype', 'g4']].sort_values(by = 'g4')
    #g4_new_order['grad'] = list(range(0,N))
    g4_new_order['grad'] = grad_nums
    g4_new_order['index_new'] =  list(range(0,N))
    
    g5_new_order = sub_grads[['index', 'Subtype', 'g5']].sort_values(by = 'g5')
    #g5_new_order['grad'] = list(range(0,N))
    g5_new_order['grad'] = grad_nums
    g5_new_order['index_new'] =  list(range(0,N))
    
    return g1_new_order, g2_new_order, g3_new_order, g4_new_order, g5_new_order

'''


# In[ ]:


old_proccess_grads = '''pull_list = []
for i in list(range(0,len(sub_class_list)*4, 4)):
    pull_list.append(list(np.array(list(range(0, 4, 1))) + i))
    
def extract_grads(grad_fit, num):
    
        rc_g1=grad_fit.gradients_[num][:,0]
        rc_g2=grad_fit.gradients_[num][:,1]
        rc_g3=grad_fit.gradients_[num][:,2]
        rc_g4=grad_fit.gradients_[num][:,3]
        rc_g5=grad_fit.gradients_[num][:,4]
    
        Y = np.stack((rc_g1, rc_g2, rc_g3, rc_g4, rc_g5)).T
        
        return Y
    

for i,j in zip(range(len(sub_class_list)), pull_list):
    
    sub_class_list[i].main_grads_processed = process_grads(grads, sub_class_list[i].df1, j[0], 
                                                           sub_class_list[i].colors, 
                                                           filepath= sub_class_list[i].plot_path+'_main_')
    
    
    sub_class_list[i].replace_grads_processed = process_grads(grads, sub_class_list[i].df1, j[1],
                                                              sub_class_list[i].colors, 
                                                              filepath=sub_class_list[i].plot_path+'_replace_')
    
    sub_class_list[i].suppress_grads_processed = process_grads(grads, sub_class_list[i].df1, j[2], 
                                                               sub_class_list[i].colors, 
                                                               sub_class_list[i].plot_path+'_suppress_')
    
    sub_class_list[i].clear_grads_processed = process_grads(grads, sub_class_list[i].df1, j[3], 
                                                            sub_class_list[i].colors, 
                                                            sub_class_list[i].plot_path+'_clear_')
'''


# In[ ]:


def create_new_orders(data):
    
    main = data.main_grads_processed
    replace = data.replace_grads_processed
    suppress = data.suppress_grads_processed
    clear = data.clear_grads_processed
    
    g1_new_orders=[]
    g2_new_orders=[]
    g3_new_orders=[]
    for i in main, replace, suppress, clear:
  
        N = 360

        g1 = i[0]
        g2 = i[1]
        g3 = i[2]

        grad_color_maps_r = get_color_maps('jet_r', N, int(N/N))
        grad_color_maps = get_color_maps('jet', N, int(N/N))
        # add color maps to gradient dfs
        g1_new_order = pd.merge(g1, grad_color_maps, on='index_new')
        g2_new_order = pd.merge(g2, grad_color_maps, on='index_new')
        g3_new_order = pd.merge(g3, grad_color_maps, on='index_new')
        
        op_names = i[0].name

        g1_new_order['ops'] = op_names
        g1_new_order['gradient'] = 1
        g1_new_order.name = op_names

        g2_new_order['ops'] = op_names
        g2_new_order['gradient'] = 2
        g2_new_order.name = op_names

        g3_new_order['ops'] = op_names
        g3_new_order['gradient'] = 3
        g3_new_order.name = op_names
        
        g1_new_orders.append(g1_new_order)
        g2_new_orders.append(g2_new_order)
        g3_new_orders.append(g3_new_order)
    
    grad1_all_ops = pd.concat(g1_new_orders)
    grad2_all_ops = pd.concat(g2_new_orders)
    grad3_all_ops = pd.concat(g3_new_orders)
    
    return grad1_all_ops, grad2_all_ops, grad3_all_ops


# In[ ]:


def grouped_grads(data):

    grad_ops_list = []
    grad_grad_list = []
    grad_sub_list = []

    grouped_grad = data.groupby(['ops', 'grad', 'Subtype']).count()

    for i in range(0,grouped_grad.shape[0], 1):
        grad_ops_list.append(grouped_grad.index[i][0])
        grad_grad_list.append(grouped_grad.index[i][1])
        grad_sub_list.append(grouped_grad.index[i][2])


    grouped_grad = grouped_grad.reset_index(drop=True)[['index']].rename({'index':'count'},axis=1)
    grouped_grad['ops'] = grad_ops_list
    grouped_grad['grads'] = grad_grad_list
    grouped_grad['subs'] = grad_sub_list
    
    return(grouped_grad)

#grouped_grad_1 = grouped_grads(grad1_all_ops)
#grouped_grad_2 = grouped_grads(grad2_all_ops)
#grouped_grad_3 = grouped_grads(grad3_all_ops)


# In[ ]:


#show_grads(sub_class_list[0].grad1_all_ops.query('ops == "clear"'))


# In[ ]:


for i in range(len(sub_class_list)):
    for j in range(3):
        sub_class_list[i].main_grads_processed[j].name = 'main'
        sub_class_list[i].replace_grads_processed[j].name = 'replace'
        sub_class_list[i].suppress_grads_processed[j].name = 'suppress'
        sub_class_list[i].clear_grads_processed[j].name = 'clear'


# In[ ]:


for i in range(len(sub_class_list)):
    sub_class_list[i].grad1_all_ops, sub_class_list[i].grad2_all_ops, sub_class_list[i].grad3_all_ops = create_new_orders(sub_class_list[i])
    sub_class_list[i].grouped_grad_1  = grouped_grads(sub_class_list[i].grad1_all_ops)
    sub_class_list[i].grouped_grad_2 = grouped_grads(sub_class_list[i].grad2_all_ops)
    sub_class_list[i].grouped_grad_3  = grouped_grads(sub_class_list[i].grad3_all_ops)


# In[ ]:


from PIL import Image, ImageColor

cmap = cm.get_cmap('jet', 20)    # PiYG
converted_list = []
for i in range(cmap.N):
    rgba = cmap(i)
    # rgb2hex accepts rgb or rgba
    hexc = matplotlib.colors.rgb2hex(rgba)
    codes = ImageColor.getcolor(str(hexc), "RGB")
    converted_list.append(hexc)

#converted_list


# In[ ]:


def bin_grad_plots(data, col, filepath=None):
    
    fig = plt.figure(figsize=(30, 16))

    sns.set_theme(style="whitegrid")

    g = sns.catplot(
        data=data, kind="violin",
        x="grad", y=col, hue="grad",
        col="ops",
        #col_wrap = 4,
        palette=jets, 
        #cmap = converted_list,
        alpha=.6, 
        height=6
    )
    
    (g.set_axis_labels("Parcel Bins", "Gradient Score",  weight='bold')
          .set_titles("{col_name}", weight='bold')
          .despine(left=True))  
    
    g.fig.savefig(f'{filepath}.png')
    #plt.clf()
    return g
    


# In[ ]:


for i in range(len(sub_class_list)):
    bin_grad_plots(sub_class_list[i].grad1_all_ops, 'g1', sub_class_list[i].plot_path+'g1_bins')
    bin_grad_plots(sub_class_list[i].grad2_all_ops, 'g2', sub_class_list[i].plot_path+'g2_bins')
    bin_grad_plots(sub_class_list[i].grad3_all_ops, 'g3', sub_class_list[i].plot_path+'g3_bins')


# In[ ]:


def grad_cor_mat(grad_data, ops_df, threshold=None, cosine=None,  filepath=None):
    
    op = ops_df.name

    test = grad_data.query('ops == '+'"'+op+'"'+'')[['index', 'grad']]
    test_merge = pd.merge(test, ops_df.reset_index(), on = 'index').sort_values('grad').drop('index', axis=1)#.reset_index(drop=True)

    X = np.array(test_merge)
    X_Z = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, dtype=np.float64, ddof=1, keepdims=True)
    D, rho = spearmanr(np.array(X_Z), axis=1)
    
    if threshold is not None:
        perc = np.array([np.percentile(x, 90) for x in D])

        for i in range(D.shape[0]):
            D[i, D[i,:] < perc[i]] = 0  
            
    
    if cosine is not None:

        D = 1 - pairwise_distances(D, metric = 'cosine')

    g = sns.clustermap(D, 
                           row_colors= get_colors('jet', 360),
                           col_colors=get_colors('jet', 360),
                           metric = 'cosine',
                           row_cluster=False, col_cluster=False, 
                           cmap = 'seismic',
                           center=0,
                           vmin=-1, vmax=1,
                           linewidths=0, xticklabels=False, yticklabels=False,
                           #**kws
                          )
    if filepath is not None:
        g.fig.savefig(f'{filepath}.png')
        
    plt.clf()
    


# In[ ]:


for i in range(len(sub_class_list)):
    for j,n in zip([sub_class_list[i].main, sub_class_list[i].replace,  sub_class_list[i].suppress,  sub_class_list[i].clear],
                   ['main', 'replace', 'suppress', 'clear']):
        
        grad_cor_mat(sub_class_list[i].grad1_all_ops, j, 
                     filepath=sub_class_list[i].plot_path+n+'_cormat')


# In[ ]:


def grad_means(data, ops_df, op):

    test = data.query('ops == '+'"'+op+'"'+'')[['index', 'grad']]
    test_merge = pd.merge(test, ops_df.reset_index(), on = 'index').drop('index', axis=1)

    #cor_mat = tester_merge.groupby('grad').mean().T.corr().reset_index(drop=True)
    #sns.clustermap(cor_mat,row_colors=jets,col_colors=jets,cmap = 'seismic')

    merged_mean = test_merge.groupby('grad').mean().reset_index(drop=True).T

    final_mean = pd.DataFrame(merged_mean.mean()).rename({0:'mean'}, axis=1)
    final_mean['grad'] = list(range(1,21,1))
    final_mean['ops'] = op
    
    return final_mean

def get_gradient_means(data):

#sns.lineplot(data=xmean, x="grad", y="mean")
    g1_rsa_ops_means = []
    g2_rsa_ops_means = []
    g3_rsa_ops_means = []

    for i in ["main", "replace", "suppress", "clear"]:

        if i == "main":
            ops_df = data.main

        elif i == "replace":
            ops_df = data.replace

        elif i == "suppress":
            ops_df = data.suppress

        elif i == "clear":
            ops_df = data.clear

        output1 = grad_means(data.grad1_all_ops, ops_df, i)
        output2 = grad_means(data.grad2_all_ops, ops_df, i)
        output3 = grad_means(data.grad3_all_ops, ops_df, i)

        g1_rsa_ops_means.append(output1)
        g2_rsa_ops_means.append(output2)
        g3_rsa_ops_means.append(output3)

    g1_rsa_ops_means_2 = pd.concat(g1_rsa_ops_means).reset_index(drop=True)
    g2_rsa_ops_means_2 = pd.concat(g2_rsa_ops_means).reset_index(drop=True)
    g3_rsa_ops_means_2 = pd.concat(g3_rsa_ops_means).reset_index(drop=True)
    
    data.g1_rsa_ops_means_2 = g1_rsa_ops_means_2
    data.g2_rsa_ops_means_2 = g2_rsa_ops_means_2
    data.g3_rsa_ops_means_2 = g3_rsa_ops_means_2
    
    return  data


# In[ ]:


sub_class_list = *map(get_gradient_means, sub_class_list),


# In[ ]:


for i in range(len(sub_class_list)):
    
    fig = plt.figure(figsize=(10, 3))
    sns.set_theme(style="darkgrid")
    g = sns.lineplot(data=sub_class_list[i].g1_rsa_ops_means_2, x="grad", y="mean", hue = "ops", palette = ops_cols)
    #fig.savefig(sub_class_list[i].plot_path+j+'_g1_rsa_means.png')
    plt.clf()
    fig2 = plt.figure(figsize=(10, 3))
    g2 = sns.lineplot(data=sub_class_list[i].g2_rsa_ops_means_2, x="grad", y="mean", hue = "ops", palette = ops_cols)
    #fig2.savefig(sub_class_list[i].plot_path+j+'_g2_rsa_means.png')
    plt.clf()
    fig3 = plt.figure(figsize=(10, 3))
    g3 = sns.lineplot(data=sub_class_list[i].g3_rsa_ops_means_2, x="grad", y="mean", hue = "ops", palette = ops_cols)
    #fig3.savefig(sub_class_list[i].plot_path+j+'_g3_rsa_means.png')
    plt.clf()


# In[ ]:


def sub_grad_plots(data, col, filepath):
    
    fig = plt.figure(figsize=(30, 16))

    sns.set_theme(style="whitegrid")

    g = sns.catplot(
        data=data, kind="violin",
        x="Subtype", y=col, hue="Subtype",
        col="ops",
        #col_wrap = 4,
        palette=net_cols, 
        alpha=.2, 
        height=6
    )
    
    (g.set_axis_labels("Network", "Gradient Score",  weight='bold')
      .set_titles("{col_name}", weight='bold')
      .despine(left=True))  

    if filepath is not None:
        g.fig.savefig(f'{filepath}.png')
    
    return g
    
#sub_grad_plots(grad1_all_ops, 'g1')
#sub_grad_plots(grad2_all_ops, 'g2')
#sub_grad_plots(grad3_all_ops, 'g3')


# In[ ]:


for i in range(len(sub_class_list)):
    fig = plt.figure(figsize=(10, 3))
    sns.set_theme(style="darkgrid")
    g = sub_grad_plots(sub_class_list[i].grad1_all_ops, 'g1', sub_class_list[i].plot_path+'nets_g1')
    plt.clf()
    fig = plt.figure(figsize=(10, 3))
    sns.set_theme(style="darkgrid")
    g = sub_grad_plots(sub_class_list[i].grad2_all_ops, 'g2', sub_class_list[i].plot_path+'nets_g2')
    plt.clf()
    fig = plt.figure(figsize=(10, 3))
    sns.set_theme(style="darkgrid")
    g = sub_grad_plots(sub_class_list[i].grad3_all_ops, 'g3', sub_class_list[i].plot_path+'nets_g3')
    plt.clf()


# In[ ]:


def comb_grads(data, op):
    
    x1 = data.grad1_all_ops.query('ops == '+'"'+op+'"'+'')[['index', 'Subtype', 'g1']]
    x2 = data.grad2_all_ops.query('ops == '+'"'+op+'"'+'')[['index', 'g2']]
    x3 = data.grad3_all_ops.query('ops == '+'"'+op+'"'+'')[['index', 'g3']]

    merged_grads = pd.merge(x1, x2, on ='index')
    merged_grads = pd.merge(merged_grads, x3, on ='index').sort_values('index').reset_index(drop=True)
    
    if op == "main":
        data.main_grads = merged_grads 
    elif op == "replace":
        data.replace_grads = merged_grads 
    elif op == "suppress":
        data.suppress_grads = merged_grads 
    elif op == "clear":
        data.clear_grads = merged_grads 
    
    return data

def get_all_grad_ops(data, g):
    
    grad_all = pd.merge(data.main_grads[['index', g]], data.replace_grads[['index', g]], on = 'index')
    grad_all = pd.merge(grad_all, data.suppress_grads[['index', g]], on = 'index')
    grad_all = pd.merge(grad_all, data.clear_grads[['index', g]], on = 'index')
    grad_all.columns = ['index', 'maintain', 'replace', 'suppress', 'clear']
    grad_all_melt = grad_all.melt(id_vars=['index']).rename({'variable':'ops', 'value':'gradient'}, axis =1)
    
    if g == 'g1':
        data.g1_all = grad_all
        data.g1_all_melt = grad_all_melt
    elif g == 'g2':
        data.g2_all = grad_all
        data.g2_all_melt = grad_all_melt
    elif g == 'g3':
        data.g3_all = grad_all
        data.g3_all_melt = grad_all_melt

    return data


# In[ ]:


for j in ['main', 'replace', 'suppress', 'clear']:
    sub_class_list = *map(partial(comb_grads, op=j), sub_class_list),
    
for j in ['g1', 'g2', 'g3']:
    sub_class_list = *map(partial(get_all_grad_ops, g=j), sub_class_list),


# In[ ]:


old_loop='''
for i in range(len(sub_class_list)):
    for j in ["main", "replace", "suppress", "clear"]:
        sub_class_list[i] =  comb_grads(sub_class_list[i], j)

for i in range(len(sub_class_list)):
    for j in ['g1', 'g2', 'g3']:
        sub_class_list[i] =  get_all_grad_ops(sub_class_list[i], j)
'''


# In[ ]:


for i in range(len(sub_class_list)):
    for j,k in zip([sub_class_list[i].g1_all_melt, 
                    sub_class_list[i].g2_all_melt, 
                    sub_class_list[i].g3_all_melt], 
                   ['g1', 'g2', 'g3']):
        
        sns.set_style("darkgrid")
        g = sns.displot(data=j, x='gradient',hue='ops', kind='kde',
                   linewidth=3, palette=ops_cols)
        
        (g.set_axis_labels("Gradient Score", "Density",  weight='bold')
          .set_titles("{col_name}", weight='bold')
          .despine(left=True))  
            
        g.tight_layout()
        g.savefig(sub_class_list[i].plot_path+k+'_hist.png')
        
        plt.clf()
    


# In[ ]:


for i in range(len(sub_class_list)):
    sub_class_list[i].main_grads.name = 'main'  
    sub_class_list[i].replace_grads.name = 'replace'
    sub_class_list[i].suppress_grads.name = 'suppress'
    sub_class_list[i].clear_grads.name = 'clear'


# In[ ]:


def get_dist(data, op, node_list, squared):
    
    if op == 'main':
        new_data = data.main_grads[['g1', 'g2', 'g3']]
        new_data.name = data.main_grads.name
        
    if op == 'replace':
        new_data = data.replace_grads[['g1', 'g2', 'g3']]
        new_data.name = data.replace_grads.name
        
    if op == 'suppress':
        new_data = data.suppress_grads[['g1', 'g2', 'g3']]
        new_data.name = data.suppress_grads.name
            
    if op == 'clear':
        new_data = data.clear_grads[['g1', 'g2', 'g3']]
        new_data.name = data.clear_grads.name
    
    from sklearn.metrics.pairwise import euclidean_distances
    euc = euclidean_distances(new_data, squared=squared)
    euc_copy = euc.copy()
    tril = np.triu_indices(len(euc))
    euc[tril] = np.nan
    eucm = pd.DataFrame(euc).melt().dropna().reset_index(drop=True)

    names=[]
    nodes1=[]
    nodes2=[]
    for name in itertools.combinations(node_list,2):
        node1 = name[0]
        node2 = name[1]
        names.append(name)
        nodes1.append(node1)
        nodes2.append(node2)

    eucm['pair'] = names
    eucm['node1'] = nodes1
    eucm['node2'] = nodes2
    
    if new_data.name == 'main':
        data.main_dist_mask = euc_copy
        data.main_dist =  eucm
        
    if new_data.name == 'replace':
        data.replace_dist_mask = euc_copy
        data.replace_dist =  eucm
        
    if new_data.name == 'suppress':
            data.suppress_dist_mask = euc_copy
            data.suppress_dist =  eucm
            
    if new_data.name == 'clear':
            data.clear_dist_mask = euc_copy
            data.clear_dist =  eucm
            
    return data


# In[ ]:


for z in ["main", "replace", "suppress", "clear"]:
     sub_class_list = *map(partial(get_dist, op=z, node_list = range(360), squared=False), sub_class_list),


# In[ ]:


def distance_additions(data):
    data.clear_suppress_dist = data.clear_dist['value'] - data.suppress_dist['value']
    data.clear_replace_dist = data.clear_dist['value'] - data.replace_dist['value']
    data.clear_main_dist = data.clear_dist['value'] - data.main_dist['value']

    data.suppress_replace_dist = data.suppress_dist['value'] - data.replace_dist['value']
    data.suppress_main_dist = data.suppress_dist['value'] - data.main_dist['value']

    data.replace_main_dist = data.replace_dist['value'] - data.main_dist['value']
    
    data.clear_dist_df = pd.DataFrame(data.clear_dist['value']).rename({'value':'clear_dist'}, axis=1)
    data.suppress_dist_df = pd.DataFrame(data.suppress_dist['value']).rename({'value':'suppress_dist'}, axis=1)
    data.replace_dist_df = pd.DataFrame(data.replace_dist['value']).rename({'value':'replace_dist'}, axis=1)
    data.main_dist_df = pd.DataFrame(data.main_dist['value']).rename({'value':'main_dist'}, axis=1)

    data.op_dist_all = pd.concat([data.clear_dist_df, data.suppress_dist_df, 
                                  data.replace_dist_df, data.main_dist_df], axis =1).melt()
    data.op_dist_all.columns = ['ops', 'dist']
    
    return data


# In[ ]:


sub_class_list = *map(distance_additions, sub_class_list),

for i in range(len(sub_class_list)):
    sub_class_list[i].dist_aov = getF(sub_class_list[i].op_dist_all, 'dist', 'ops').round(3)
    sub_class_list[i].dist_post_hoc = getF(sub_class_list[i].op_dist_all, 'dist', 'ops').round(3)


# In[ ]:


#for i in range(len(sub_class_list)):
#        sub_class_list[i] = distance_additions(sub_class_list[i])
#        sub_class_list[i].dist_aov = getF(sub_class_list[i].op_dist_all, 'dist', 'ops').round(3)
#        sub_class_list[i].dist_post_hoc = getF(sub_class_list[i].op_dist_all, 'dist', 'ops').round(3)


# In[ ]:


for i in range(len(sub_class_list)):

    sns.set_style("darkgrid")
    g = sns.displot(data=sub_class_list[i].op_dist_all, x='dist',hue='ops', kind='kde', palette=ops_cols, linewidth=3)
    
    (g.set_axis_labels("Parcel Pairwise Centrality", "Density",  weight='bold')
          .set_titles("{col_name}", weight='bold')
          .despine(left=True))  
    g.tight_layout()
    g.savefig(sub_class_list[i].plot_path+'paircent_all_ops_hist.png')
        
    plt.clf()

# In[ ]:


sub_class_list = list(sub_class_list)


# In[ ]:


sub_class_list_copy = sub_class_list.copy()


# In[ ]:


def node_centrality(data, node_list):
    cent_outs = []
    for i in node_list:
        t1 = data[data['node1'] == i]
        t2 = data[data['node2'] == i]
        frame = pd.concat([t1, t2])['value'].mean()
        cent_outs.append(frame)

    cent_frame = pd.DataFrame(cent_outs).rename({0:'centrality'}, axis=1)
    
    return cent_frame

def add_op_centraility(data):
    data.main_centrality = node_centrality(data.main_dist, range(360))
    data.replace_centrality = node_centrality(data.replace_dist, range(360))
    data.suppress_centrality = node_centrality(data.suppress_dist, range(360))
    data.clear_centrality = node_centrality(data.clear_dist, range(360))

    data.all_centraility = pd.concat([data.main_centrality, data.replace_centrality, 
                                 data.suppress_centrality, data.clear_centrality], axis=1)

    data.all_centraility.columns = ['main', 'replace', 'suppress', 'clear']
    
    return data

def move_op_centrality(data):
    
    data.main_grads['centrality'] = data.main_centrality #* -1
    data.replace_grads['centrality'] = data.replace_centrality #* -1
    data.suppress_grads['centrality'] = data.suppress_centrality #* -1
    data.clear_grads['centrality'] = data.clear_centrality #* -1
    
    return data


for i in range(len(sub_class_list)):
    sub_class_list[i] = add_op_centraility(sub_class_list[i])
    
    g = sns.displot(data=sub_class_list[i].all_centraility.melt(), 
                    x='value',hue='variable', kind='kde', palette=ops_cols, linewidth=3)
    
    (g.set_axis_labels("Parcel Average Centrality", "Density",  weight='bold')
          .set_titles("{col_name}", weight='bold')
          .despine(left=True))  
    g.tight_layout()
    g.savefig(sub_class_list[i].plot_path+'avgcent_all_ops_hist.png')

    
    #g.savefig(sub_class_list[0].plot_path +'op_centrality_hist.png')
    plt.clf()
    sub_class_list[i] = move_op_centrality(sub_class_list[i])


# In[ ]:


def network_cent_difs(data):

    network_cent_aovs=[]
    network_cent_difs=[]

    for i in data.main_grads, data.replace_grads, data.suppress_grads, data.clear_grads:

        aov_c = pd.DataFrame(getF(i, 'centrality', 'Subtype')).T.round(3)
        aov_c.columns = ['F', 'p']

        post_hocs_c = pd.DataFrame(getposthoc(i, 'centrality', 'Subtype')).round(3)
        post_hocs_c.columns = post_hocs_c.index
        #post_hocs_c['op'] = ['main', 'suppress', 'suppress', 'clear']

        network_cent_aovs.append(aov_c)
        network_cent_difs.append(post_hocs_c)

    data.network_cent_aov_out = pd.concat(network_cent_aovs)
    data.network_cent_aov_out['op'] = ['main', 'suppress', 'suppress','clear']


    data.network_cent_difs_out = pd.concat(network_cent_difs)
    data.network_cent_difs_out['op'] = list(itertools.chain.from_iterable(
        [['main']*4,['replace']*4, ['suppress']*4, ['clear']*4]))

    return data


# In[ ]:


for i in range(len(sub_class_list)):
    sub_class_list[i] = network_cent_difs(sub_class_list[i])


# In[ ]:


sub_class_list = list(sub_class_list)


# In[ ]:


def plot_3D_centraility(data, filename=None):
    
    op_name = data.name
        
    newX = np.array(data[['g1', 'g2', 'g3', 'centrality']])

    fig = plt.figure()

    data = go.Scatter3d(x=newX[:,0], y=newX[:,1], z=newX[:,2], 
                        mode='markers',
                        marker=dict(size=5,
                                    color=newX[:,3],
                                    opacity=0.7,
                                    colorscale='inferno')
                       )

    layout = go.Layout(title_text=op_name,title_x=0.5,title_y=0.8,title_font_size=20)
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

    if filename is not None:
        fig.write_html(filename+op_name+'_centrality.html')
        
    
for j in range(len(sub_class_list)): 
    for i in sub_class_list[j].main_grads, sub_class_list[j].replace_grads, sub_class_list[j].suppress_grads, sub_class_list[j].clear_grads:

        plot_3D_centraility(i, sub_class_list[j].plot_path)
        #sns.set_style("darkgrid")    
        g = sns.displot(data=i, x= 'centrality', hue='Subtype', kind='kde', palette=net_cols, linewidth=3)

        g.fig.suptitle(i.name, weight='bold')
        (g.set_axis_labels("Network Centrality", "Density",  weight='bold')
          #.set_titles({"i.name"}, weight='bold')
          .despine(left=True))  
        g.tight_layout()
        g.savefig(sub_class_list[j].plot_path+i.name+'_network_cent_all_ops_hist.png')
        plt.clf()

    

# In[ ]:


parcel_centraility = '''
- For each parcel, centrality was calculated as the average Euclidean distance to all other parcels 
in the 3D gradient space (accounting for the full 3D space and not one gradient only).

- In this context, high centrality refers to the smallest distance to all other parcels in space 
(i.e. towards the center of the 3D gradient space), and thus indicates a functional connectivity 
profile that isn't differentiated across all three gradients.
'''

within_network_dispersion = '''
- Within-network dispersion was calculated as the sum of the squared Euclidean distances in the 3D 
gradient space of all parcels within that network to the network centroid (i.e., the mean coordinates
in 3D gradient space of all parcels belonging to that network). 

- A small dispersion value could be interpreted as a highly integrated network, segregated from 
other networks. These multi-dimensional gradient metrics are motivated by prior related work on 
network integration and segregation, are assumed to reflect segregation of functional networks, 
and have been demonstrated to be comparable with other approaches of measuring network changes 
such as clustering, as well as within-network connectivity and segregation (Bethlehem et al., 2020).

- Particularly, Euclidean distance in the 3D gradient space reflects the similarity of connectivity
profiles between cortical parcellations, across multiple axes of differentiation. 
'''


# In[ ]:


def sub_op_centrality(data, op):
        
    if op == 'main':
        new_data = data.main_grads
    elif op == 'replace':
        new_data = data.replace_grads
    elif op == 'suppress':
        new_data = data.suppress_grads     
    elif op == 'clear':
        new_data = data.clear_grads
    
    sub_dis=[]
    sub_dis_mask = []
    sub_centrality = []
    
    for i in [1,2,3,4]:
        test = new_data.query('Subtype =='+str(i))
        index_lists = test['index'].to_list()
        test_grads = test[['g1', 'g2', 'g3']]
        test_mean = test_grads.mean()

        dis_out=[]
        for j in index_lists:
            filtered = test.query('index =='+str(j))[['g1', 'g2', 'g3']]
            l = pd.concat([filtered,pd.DataFrame(test_mean).T])
            #print(l)
            euc = np.array(euclidean_distances(l, squared=True)[0][1])
            dis_out.append(euc)
            
        dispersion = pd.DataFrame(dis_out)
        dispersion['index'] = index_lists
        #dispersion_mask, dispersion = get_dist(test, index_lists, True)
        #sub_cents = node_centrality(dispersion, index_lists)
        dispersion['Subtype'] = i
        dispersion['op'] = op
        
        #sub_centrality.append(sub_cents)
        sub_dis.append(dispersion)
        #sub_dis_mask.append(dispersion_mask)
    
    sub_dispersion = pd.concat(sub_dis).reset_index(drop=True).rename({0:'dispersion'}, axis = 1)
    
    if op == 'main':
        data.main_op_disp = sub_dispersion
    elif op == 'replace':
        data.replace_op_disp = sub_dispersion
    elif op == 'suppress':
        data.suppress_op_disp = sub_dispersion     
    elif op == 'clear':
        data.clear_op_disp = sub_dispersion
        
    return data
        
    
#main_op_disp = sub_op_centrality(main_grads, 'maintain')  
#replace_op_disp = sub_op_centrality(replace_grads, 'replace')  
#suppress_op_disp =  sub_op_centrality(suppress_grads, 'suppress')  
#clear_op_disp = sub_op_centrality(clear_grads, 'clear')  


# In[ ]:


for i in range(len(sub_class_list)):
    for z in ['main', 'replace', 'suppress', 'clear']:
        sub_class_list[i] = sub_op_centrality(sub_class_list[i], z)


# In[ ]:


def network_disp_difs(data):

    network_disp_aovs=[]
    network_disp_difs=[]

    for i in range(1,5,1):
        sub_disp_frame = pd.concat([
           data.main_op_disp.query('Subtype =='+str(i)),
           data.replace_op_disp.query('Subtype == '+str(i)),
           data.suppress_op_disp.query('Subtype == '+str(i)),
           data.clear_op_disp.query('Subtype == '+str(i))
        ])

        aov = pd.DataFrame(getF(sub_disp_frame, 'dispersion', 'op')).T.round(3)
        aov.columns = ['F', 'p']
        aov['Subtype'] = i

        post_hocs = pd.DataFrame(getposthoc(sub_disp_frame, 'dispersion', 'op')).round(3)
        post_hocs.columns = post_hocs.index
        post_hocs['Subtype'] = i

        network_disp_aovs.append(aov)
        network_disp_difs.append(post_hocs)

    data.network_disp_aov_out = pd.concat(network_disp_aovs)
    data.network_disp_difs_out = pd.concat(network_disp_difs)
    
    return data


# In[ ]:


for i in range(len(sub_class_list)):
    sub_class_list[i] = network_disp_difs(sub_class_list[i])


# In[ ]:


for i in range(len(sub_class_list)):
    for j,k in zip([sub_class_list[i].main_op_disp, sub_class_list[i].replace_op_disp, sub_class_list[i].suppress_op_disp, sub_class_list[i].clear_op_disp],
                   ['main', 'replace', 'suppress', 'clear']):
            
        sns.set_style("darkgrid")
        g =  sns.displot(j, x= 'dispersion', hue='Subtype', kind='kde', palette=net_cols, linewidth=3)
        
        g.fig.suptitle(k, weight='bold')
        (g.set_axis_labels("Network Dispersion", "Density",  weight='bold')
          #.set_titles({"i.name"}, weight='bold')
          .despine(left=True))  
        g.tight_layout()
        g.savefig(sub_class_list[i].plot_path+'_'+k+'_network_disp_all_ops_hist.png')
        
        plt.clf()
           # g.savefig('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/leiden/analysis/gradients/figures/dispersion/'+j+'.png')
            
    #set_ylim(0, 120)
    #sns.displot(data=i, x= 'dispersion', hue='grad1', kind='kde', palette=jets, linewidth=3)
    #sns.displot(data=i, x= 'dispersion', hue='grad2', kind='kde', palette=jets, linewidth=3)
    #sns.displot(data=i, x= 'dispersion', hue='grad3', kind='kde', palette=jets, linewidth=3)


# In[ ]:


def add_distance_frames(data):
    
    data.clear_replace_dist_df = pd.DataFrame(data.clear_replace_dist).rename({'value':'clear_replace'}, axis=1)
    data.clear_suppress_dist_df = pd.DataFrame(data.clear_suppress_dist).rename({'value':'clear_suppress'}, axis=1)
    data.clear_main_dist_df = pd.DataFrame(data.clear_main_dist).rename({'value':'clear_main'}, axis=1)

    data.suppress_main_dist_df = pd.DataFrame(data.suppress_main_dist).rename({'value':'suppress_main'}, axis=1)
    data.suppress_replace_dist_df = pd.DataFrame(data.suppress_replace_dist).rename({'value':'suppress_replace'}, axis=1)

    data.replace_main_dist_df = pd.DataFrame(data.replace_main_dist).rename({'value':'replace_main'}, axis=1)

    data.op_dists_all = pd.concat([data.clear_replace_dist_df, data.clear_suppress_dist_df, 
                                   data.clear_main_dist_df, data.suppress_main_dist_df, 
                                   data.suppress_replace_dist_df,data.replace_main_dist_df], axis=1).melt()

    data.op_dists_all.columns = ['op_comp', 'dist']

    #sns.histplot(op_dists_all, x = 'dist', hue = 'op_comp')

    #sns.displot(op_dists_all, x = 'dist', hue = 'op_comp', linewidth=2, kind='kde')

    data.dist_op_comp_aov = getF(data.op_dists_all, 'dist', 'op_comp').round(3)

    data.dist_op_comp_posthoc = getposthoc(data.op_dists_all, 'dist', 'op_comp').round(3)
    
    return data


# In[ ]:


for i in range(len(sub_class_list)):
    sub_class_list[i] = add_distance_frames(sub_class_list[i])


# In[ ]:


for i in range(len(sub_class_list)):
    g = sns.FacetGrid(sub_class_list[i].op_dists_all, col="op_comp", hue="op_comp", col_wrap=3)
    g.map(sns.distplot, "dist")
    g.set_titles(col_template="{col_name}", row_template="") 
    
    g.fig.suptitle('Operation Distance Comparisons', weight='bold')
    (g.set_axis_labels("Distance Change", "Density",  weight='bold')
     #.set_titles({"i.name"}, weight='bold')
     .despine(left=True))  
    g.tight_layout()
    g.savefig(sub_class_list[i].plot_path+'op_dist_comps_hist.png')
    
    plt.clf()
    #g.savefig("/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/leiden/analysis/gradients/figures/op_dist_comps/op_comp_all_wrapped.png")


# In[ ]:


for i in range(len(sub_class_list)):
    sns.set_style("darkgrid")
    g = sns.displot(sub_class_list[i].op_dists_all, x= 'dist', hue='op_comp', kind = 'kde', linewidth=3)
    
    g.fig.suptitle('Operation Distance Comparisons', weight='bold')
    (g.set_axis_labels("Distance Change", "Density",  weight='bold')
     #.set_titles({"i.name"}, weight='bold')
     .despine(left=True))  
    g.tight_layout()
    g.savefig(sub_class_list[i].plot_path+'all_op_dist_comps_hist.png')
    
    plt.clf()
    #g.savefig("/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/leiden/analysis/gradients/figures/op_dist_comps/op_comp_all_kde.png")
    #g = sns.displot(sub_class_list[i].op_dists_all, x= 'dist', hue='op_comp', kind = 'ecdf', linewidth=3)   
    #plt.clf()
    #g.savefig("/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/leiden/analysis/gradients/figures/op_dist_comps/op_comp_all_ecdf.png")


# In[ ]:


def dist_change(grad_data, dist_mat, parcel_index, colors,title, filepath=None):
    sns.set_style('white')
    sorted_subtype_parcels = grad_data
    color_df = pd.DataFrame(colors).reset_index().rename({'Subtype':'Networks'}, axis = 1)
    color_df.index = parcel_index
    color_df['index'] = parcel_index
    sorted_subtype_parcels = pd.merge(sorted_subtype_parcels , color_df, on ='index').sort_values('Subtype')#.reset_index(drop=True)
    sorted_subtype_parcels_list = sorted_subtype_parcels['index'].to_list()
    network_colors = sorted_subtype_parcels['Networks'].to_list()
    
    change_dist_mat = vsim(dist_mat)
    
    change_dist_mat.columns = parcel_index
    change_dist_mat.index = parcel_index
    
    change_dist_mat = change_dist_mat[sorted_subtype_parcels_list].T
    change_dist_mat = change_dist_mat[sorted_subtype_parcels_list].T
    
    #print(change_dist_mat)
    #change_dist_mat = threshold_proportional(np.array(change_dist_mat), .75)
    
    change_dist_mat = np.array(change_dist_mat)
    
    #perc = np.array([np.percentile(x, 90) for x in change_dist_mat])

    #for i in range(change_dist_mat.shape[0]):
    #    change_dist_mat[i, change_dist_mat[i,:] < perc[i]] = 0    
        

    matrix = pd.DataFrame(np.tril(change_dist_mat))
    #matrix = change_dist_mat
    
    #change_dist_mat = threshold_proportional(change_dist_mat, .15)
    
    #kws = dict(cbar_kws=dict(ticks=[-.03, 0, .03]), figsize=(6, 6))
    kws = dict(cbar_kws=dict(ticks=[-.3, 0, .1]), figsize=(6, 6))
    g = sns.clustermap(matrix, 
                       row_colors=network_colors,
                       #metric = 'cosine',
                       col_colors=network_colors,
                       row_cluster=False, col_cluster=False, 
                       cmap = 'seismic',
                       center=0,
                       #vmin=-.03, vmax=.03,
                       vmin=-.1, vmax=.1,
              
                       linewidths=0, xticklabels=False, yticklabels=False,
                       #**kws
                      )

    g.cax.set_position([.09, .17, .03, .45])

    ax_col_colors = g.ax_col_colors
    box = ax_col_colors.get_position()
    box_heatmap = g.ax_heatmap.get_position()
    ax_col_colors.set_position([.238, -.01, box.width*1, .025])
    
    g.fig.suptitle(title+' Distance Comparison', weight='bold')
    #g.savefig(sub_class_list[i].plot_path+'dist_comps_mat.png')
    
    #if filepath is not None:
    #    g.savefig('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/leiden/analysis/gradients/figures/op_dist_comps/'+filepath+'.png')
    
    #return(change_dist_mat)


# In[ ]:


#dist_change(sub_class_list[i].clear_grads, j, range(360), sub_class_list[i].colors, 'Clear-Main')


# In[ ]:


for i in range(len(sub_class_list)):
    for j,k in zip([sub_class_list[i].clear_main_dist, sub_class_list[i].clear_replace_dist,
                    sub_class_list[i].clear_suppress_dist, sub_class_list[i].suppress_main_dist, 
                    sub_class_list[i].suppress_replace_dist, sub_class_list[i].replace_main_dist],
                   ['Clear-Main','Clear-Replace', 'Clear-Suppress', 'Suppress-Main', 'Suppress-Replace', 'Replace-Main']):
        dist_change(sub_class_list[i].clear_grads, j, range(360), sub_class_list[i].colors, k)
        plt.clf()


# In[ ]:


def grad_mds(op_df, grad_df, filepath=None): 
    
    main_cor = 1-np.corrcoef(op_df)/2
    mds = MDS(n_components=3, random_state=0,n_jobs=4, dissimilarity="precomputed")
    X1 = mds.fit_transform(np.array(main_cor))
    X1_frame = pd.DataFrame(X1)

    X1_frame['grad'] = grad_df['grad']

    means = np.array(X1_frame.groupby('grad').mean().reset_index().iloc[:, 1:4])
    sds = np.array(X1_frame.groupby('grad').std().reset_index().iloc[:, 1:4])

    jets = get_colors('jet_r', 20)
    fig = plt.figure()
    data = go.Scatter3d(x=means[:,0], y=means[:,1], z=means[:,2], 
                        mode='markers',
                        marker=dict(size=10,
                                    #color=newX[:,1],
                                    color=jets,
                                    opacity=0.7,
                                    colorscale=jets)
                       )


    layout = go.Layout(title_text="",title_x=0.5,title_y=0.8,title_font_size=12)
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

    fig.show()

    if filepath is not None:
        fig.write_html(filepath+'_grad_mds.html')
        
    plt.clf()

    
#def plot_grad_mds(op_df, grad_num, colors, filepath=None):
#    for i in g1_new_orders, g2_new_orders, g3_new_orders:
#        testg = i[grad_num].sort_values('index').reset_index(drop=True)
#        grad_mds(op_df, testg, colors, filepath)


# In[ ]:


def plot_class_grad_mds(data, op):
    
    if op == 'main':
        op_df = data.main
    elif op == 'replace':
        op_df = data.replace
    elif op == 'suppress':
        op_df = data.suppress
    elif op == 'clear':
        op_df = data.clear
        
    x1= data.grad1_all_ops.query('ops == '+'"'+op+'"'+'').sort_values('index').reset_index(drop=True)
    x2 = data.grad2_all_ops.query('ops == '+'"'+op+'"'+'').sort_values('index').reset_index(drop=True)
    x3 = data.grad3_all_ops.query('ops == '+'"'+op+'"'+'').sort_values('index').reset_index(drop=True)
    
    for i in x1,x2,x3:
        grad_mds(op_df, i, data.plot_path+op)


# In[ ]:


#for j in ['main', 'replace', 'suppress', 'clear']:
#    *map(partial(plot_class_grad_mds, op=j), sub_class_list),


# In[ ]:


for i in range(len(sub_class_list)):
    sub_class_list[i].main_op_disp['sub'] = sub_class_list[i].name
    sub_class_list[i].replace_op_disp['sub'] = sub_class_list[i].name
    sub_class_list[i].suppress_op_disp['sub'] = sub_class_list[i].name
    sub_class_list[i].clear_op_disp['sub'] = sub_class_list[i].name
    
    sub_class_list[i].main_op_disp.to_csv(sub_class_list[i].plot_path+'main_network_disp.csv')
    sub_class_list[i].replace_op_disp.to_csv(sub_class_list[i].plot_path+'replace_network_disp.csv')
    sub_class_list[i].suppress_op_disp.to_csv(sub_class_list[i].plot_path+'suppress_network_disp.csv')
    sub_class_list[i].clear_op_disp.to_csv(sub_class_list[i].plot_path+'clear_network_disp.csv')
    
    sub_class_list[i].g1_all['sub'] = sub_class_list[i].name
    sub_class_list[i].g2_all['sub'] = sub_class_list[i].name
    sub_class_list[i].g3_all['sub'] = sub_class_list[i].name
    
    sub_class_list[i].g1_all.to_csv(sub_class_list[i].plot_path+'g1_all_ops.csv')
    sub_class_list[i].g2_all.to_csv(sub_class_list[i].plot_path+'g2_all_ops.csv')
    sub_class_list[i].g3_all.to_csv(sub_class_list[i].plot_path+'g3_all_ops.csv')
    


# In[ ]:


subj_accuracy = pd.concat(accuracy_list)
subj_accuracy['subj'] = sub_idx
subj_accuracy.to_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/subj/predictions/classifiers/accuracy.csv')

subj_evidence = pd.concat(evidence_list)
subj_evidence['subj'] = sub_idx
subj_evidence.to_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/subj/predictions/classifiers/evidence.csv')


# In[ ]:


copy_subs = sub_class_list.copy()

main_cent_prediction = []
replace_cent_prediction = []
suppress_cent_prediction = []
clear_cent_prediction = []

main_dist_prediction = []
replace_dist_prediction = []
suppress_dist_prediction = []
clear_dist_prediction = []

for i in range(len(copy_subs)):
    
    copy_subs[i].main_centrality_predictions = copy_subs[i].main_centrality.T
    copy_subs[i].main_centrality_predictions[['t1_maintain_accuracy']] = accuracy_list[i]['t1_maintain'][0]
    copy_subs[i].main_centrality_predictions[['t1_maintain_evidence']] = evidence_list[i]['t1_maintain'][0]
    copy_subs[i].main_centrality_predictions[['ID']] = str(copy_subs[i].name).replace('_sm_vector', "")
        
    copy_subs[i].replace_centrality_predictions = copy_subs[i].replace_centrality.T
    copy_subs[i].replace_centrality_predictions[['t2_repCat_accuracy']] = accuracy_list[i]['t2_repCat'][0]
    copy_subs[i].replace_centrality_predictions[['t2_repCat_evidence']] = evidence_list[i]['t2_repCat'][0]
    copy_subs[i].replace_centrality_predictions[['ID']] = str(copy_subs[i].name).replace('_sm_vector', "")
    
    copy_subs[i].suppress_centrality_predictions = copy_subs[i].suppress_centrality.T
    copy_subs[i].suppress_centrality_predictions[['t3_target_accuracy']] = accuracy_list[i]['t3_target'][0]
    copy_subs[i].suppress_centrality_predictions[['t3_target_evidence']] = evidence_list[i]['t3_target'][0]
    copy_subs[i].suppress_centrality_predictions[['ID']] = str(copy_subs[i].name).replace('_sm_vector', "")
    
    copy_subs[i].clear_centrality_predictions = copy_subs[i].clear_centrality.T
    copy_subs[i].clear_centrality_predictions[['t4_global_accuracy']] = accuracy_list[i]['t4_global'][0]
    copy_subs[i].clear_centrality_predictions[['t4_global_evidence']] = evidence_list[i]['t4_global'][0]
    copy_subs[i].clear_centrality_predictions[['ID']] = str(copy_subs[i].name).replace('_sm_vector', "")
    
    #----------------------------------------------------------------------------------------
    
    copy_subs[i].main_dist_predictions = copy_subs[i].main_dist_df.T
    copy_subs[i].main_dist_predictions[['t1_maintain_accuracy']] = accuracy_list[i]['t1_maintain'][0]
    copy_subs[i].main_dist_predictions[['t1_maintain_evidence']] = evidence_list[i]['t1_maintain'][0]
    copy_subs[i].main_dist_predictions[['ID']] = str(copy_subs[i].name).replace('_sm_vector', "")
    
    
    copy_subs[i].replace_dist_predictions = copy_subs[i].replace_dist_df.T
    copy_subs[i].replace_dist_predictions[['t2_repCat_accuracy']] = accuracy_list[i]['t2_repCat'][0]
    copy_subs[i].replace_dist_predictions[['t2_repCat_evidence']] = evidence_list[i]['t2_repCat'][0]
    copy_subs[i].replace_dist_predictions[['ID']] = str(copy_subs[i].name).replace('_sm_vector', "")
    
    
    copy_subs[i].suppress_dist_predictions = copy_subs[i].suppress_dist_df.T
    copy_subs[i].suppress_dist_predictions[['t3_target_accuracy']] = accuracy_list[i]['t3_target'][0]
    copy_subs[i].suppress_dist_predictions[['t3_target_evidence']] = evidence_list[i]['t3_target'][0]
    copy_subs[i].suppress_dist_predictions[['ID']] = str(copy_subs[i].name).replace('_sm_vector', "")
    
    
    copy_subs[i].clear_dist_predictions = copy_subs[i].clear_dist_df.T
    copy_subs[i].clear_dist_predictions[['t4_global_accuracy']] = accuracy_list[i]['t4_global'][0]
    copy_subs[i].clear_dist_predictions[['t4_global_evidence']] = evidence_list[i]['t4_global'][0]
    copy_subs[i].clear_dist_predictions[['ID']] = str(copy_subs[i].name).replace('_sm_vector', "")
    
    
    #----------------------------------------------------------------------------------------
    
    main_cent_prediction.append(copy_subs[i].main_centrality_predictions)
    replace_cent_prediction.append(copy_subs[i].replace_centrality_predictions)
    suppress_cent_prediction.append(copy_subs[i].suppress_centrality_predictions)
    clear_cent_prediction.append(copy_subs[i].clear_centrality_predictions)
    
    #----------------------------------------------------------------------------------------
        
    main_dist_prediction.append(copy_subs[i].main_dist_predictions)
    replace_dist_prediction.append(copy_subs[i].replace_dist_predictions)
    suppress_dist_prediction.append(copy_subs[i].suppress_dist_predictions)
    clear_dist_prediction.append(copy_subs[i].clear_dist_predictions)
    
     #----------------------------------------------------------------------------------------


# In[ ]:


main_cent_prediction = pd.concat(main_cent_prediction)
replace_cent_prediction = pd.concat(replace_cent_prediction)
suppress_cent_prediction = pd.concat(suppress_cent_prediction)
clear_cent_prediction = pd.concat(clear_cent_prediction)

main_dist_prediction = pd.concat(main_dist_prediction)
replace_dist_prediction = pd.concat(replace_dist_prediction)
suppress_dist_prediction = pd.concat(suppress_dist_prediction)
clear_dist_prediction = pd.concat(clear_dist_prediction)


# In[ ]:


save_cent_path = '/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/subj/predictions/centrality/'
save_dist_path = '/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/subj/predictions/distance/'

for i,j in zip([main_cent_prediction, replace_cent_prediction, suppress_cent_prediction, clear_cent_prediction],
              ['main_cent_prediction', 'replace_cent_prediction', 'suppress_cent_prediction', 'clear_cent_prediction']):
              i.to_csv(save_cent_path+j+'.csv')
for i,j in zip([main_dist_prediction, replace_dist_prediction, suppress_dist_prediction, clear_dist_prediction],
               ['main_dist_prediction', 'replace_dist_prediction', 'suppress_dist_prediction', 'clear_dist_prediction']):
               i.to_csv(save_dist_path+j+'.csv')


