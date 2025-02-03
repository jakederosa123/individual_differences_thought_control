#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/env/lib/python3.7/site-packages')
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/')
from functions import *

from glob import glob

def show_code(function):  
    import inspect
    lines = inspect.getsource(function)
    print(lines)  


# In[ ]:


net_cols = ['#FCFF0D', '#21DFB4', '#4E00A2', '#F00087']
ops_cols = ['#F0180A', '#F08B0A', '#6DAE45', '#0A5AF0']


# In[4]:


group_analysis = 'group_sm_vector_spearman'
data_path = '/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/leiden/'
path = data_path+group_analysis+'/Output/Results/'+group_analysis +'_Full_Subtypes.csv'
tp = pd.read_csv(path, iterator=True, chunksize=100000) 
df1 = pd.concat(tp, ignore_index=True)
df1 = df1.drop(['Unnamed: 0'], axis =1)


# In[5]:


main_grads = pd.read_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/grp/main_grads.csv')
replace_grads = pd.read_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/grp/replace_grads.csv')
suppress_grads = pd.read_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/grp/suppress_grads.csv')
clear_grads = pd.read_csv('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/grp/clear_grads.csv')


# In[6]:


colors = df1.Subtype.map({1:net_cols[0], 2:net_cols[1], 3:net_cols[2], 4:net_cols[3]})


# In[7]:


def grad3d(grad, gradient=None, outpath=None):
    
    global df1 
    
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    sns.set(style='white', rc={'figure.figsize':(10,7)})

    fig = plt.figure(dpi = 300)
    ax = fig.add_subplot(111, projection='3d')
    
    if gradient is not None:
        scat = ax.scatter3D(grad.iloc[:,1], grad.iloc[:,2], zs=grad.iloc[:,3], alpha = 0.8, c =  grad.iloc[:,2], cmap = 'jet')  
    else: 
        scat = plt.scatter(grad.iloc[:,1], grad.iloc[:,2], zs=grad.iloc[:,3], s=30, c = colors, alpha=0.7)
 
    plt.et_edgecolors = plt.set_facecolors = lambda *args:None
    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.view_init(10, 205)

    # Get rid of colored axes planes
    # First remove fill
    pane_rbga = (239/255, 239/255, 239/255, 255/255)
    ax.w_xaxis.set_pane_color(pane_rbga)
    ax.w_yaxis.set_pane_color(pane_rbga)
    ax.w_zaxis.set_pane_color(pane_rbga)

    ax.grid(False)

    plt.draw()
    plt.tight_layout()

    def init():
        ax.view_init(elev=10., azim=0)
        return [fig]

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return [fig]


    if outpath is not None:
        ## Animate
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True)
        # Save
        writergif = animation.PillowWriter(fps=30)

        anim.save(outpath+'.gif',writer=writergif, savefig_kwargs={'facecolor':'white'})


# In[10]:


def network_dispersion(grad, outpath=None):
    
    global df1 
    
    sns.set(style='white', rc={'figure.figsize':(10,7)})

    fig = plt.figure(dpi = 300)
    ax = fig.add_subplot(111, projection='3d')

    vn_index = df1[df1['Subtype'] == 1].index.to_list()
    smn_index = df1[df1['Subtype'] == 2].index.to_list()
    fpcn_index = df1[df1['Subtype'] == 3].index.to_list()
    dmn_index = df1[df1['Subtype'] == 4].index.to_list()

    vn_mds = pd.DataFrame(grad.iloc[:, 1:]).T[vn_index].T
    smn_mds = pd.DataFrame(grad.iloc[:, 1:]).T[smn_index].T
    fpcn_mds = pd.DataFrame(grad.iloc[:, 1:]).T[fpcn_index].T
    dmn_mds = pd.DataFrame(grad.iloc[:, 1:]).T[dmn_index].T

    vn_des = vn_mds.describe().T[['mean']]
    smn_des = smn_mds.describe().T[['mean']]
    fpcn_des = fpcn_mds.describe().T[['mean']]
    dmn_des = dmn_mds.describe().T[['mean']]

    all_x_mean = np.array([vn_des['mean'][0], smn_des['mean'][0], fpcn_des['mean'][0], dmn_des['mean'][0]]).flatten()
    all_y_mean = np.array([vn_des['mean'][1], smn_des['mean'][1], fpcn_des['mean'][1], dmn_des['mean'][1]]).flatten()
    all_z_mean = np.array([vn_des['mean'][2], smn_des['mean'][2], fpcn_des['mean'][2], dmn_des['mean'][2]]).flatten()

    all_means = np.array([all_x_mean, all_y_mean, all_z_mean]).T

    fx = all_means[:,0]
    fy = all_means[:,1]
    fz = all_means[:,2] 
                                  
    scat = plt.scatter(grad.iloc[:,1], grad.iloc[:,2], zs=grad.iloc[:,3], s=30, c = colors, alpha=0.7)
    scat_means = plt.scatter(all_means[:,0], all_means[:,1], zs=all_means[:,2],
                             s=60, c = net_cols, alpha=1,  edgecolors='black')


    # Add lines from VN parcels to VN mean
    vn_mask = df1['Subtype'] == 1
    vn_coords = grad.iloc[np.where(vn_mask)[0], 1:4].values
    vn_mean = all_means[0]
    for coord in vn_coords:
        line = np.array([coord, vn_mean])
        ax.plot(line[:,0], line[:,1], line[:,2],
                c=net_cols[0], 
                #c='black',
                #linestyle='--',
                #dashes=(5, 10),
                alpha=0.5, lw=.8)

    # Add lines from SMN parcels to SMN mean
    smn_mask = df1['Subtype'] == 2
    smn_coords = grad.iloc[np.where(smn_mask)[0], 1:4].values
    smn_mean = all_means[1]
    for coord in smn_coords:
        line = np.array([coord, smn_mean])
        ax.plot(line[:,0], line[:,1], line[:,2], 
                c=net_cols[1], 
                #linestyle='--',
                #dashes=(5, 10),
                #c='black',
                alpha=0.5, lw=.8)
        
     # Add lines from FPCN parcels to FPCN mean
    fpcn_mask = df1['Subtype'] == 3
    fpcn_coords = grad.iloc[np.where(fpcn_mask)[0], 1:4].values
    fpcn_mean = all_means[2]
    for coord in fpcn_coords:
        line = np.array([coord, fpcn_mean])
        ax.plot(line[:,0], line[:,1], line[:,2], 
                c=net_cols[2], 
                #linestyle='--',
                #dashes=(5, 10),
                #c='black',
                alpha=0.5, lw=.8)

    # Add lines from DMN parcels to DMN mean
    dmn_mask = df1['Subtype'] == 4
    dmn_coords = grad.iloc[np.where(dmn_mask)[0], 1:4].values
    dmn_mean = all_means[3]
    for coord in dmn_coords:
        line = np.array([coord, dmn_mean])
        ax.plot(line[:,0], line[:,1], line[:,2], 
                c=net_cols[3], 
                #c='black',
                #linestyle='--',
                #dashes=(5, 10),
                alpha=.5, lw=.8,
               )
        

    #scat = plt.scatter(0, 0, zs=0, s=150, color = "grey", edgecolors='black')
    plt.et_edgecolors = plt.set_facecolors = lambda *args:None
    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.view_init(10, 205)

    # Get rid of colored axes planes
    # First remove fill
    pane_rbga = (239/255, 239/255, 239/255, 255/255)

    ax.w_xaxis.set_pane_color(pane_rbga)
    ax.w_yaxis.set_pane_color(pane_rbga)
    ax.w_zaxis.set_pane_color(pane_rbga)

    import matplotlib.cm as cm

    ax.grid(False)

    plt.draw()
    plt.tight_layout()

    def init():
        ax.view_init(elev=10., azim=0)
        return [fig]

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return [fig]
    
    #ax.set_facecolor((1,1,1,1))

    if outpath is not None:
        ## Animate
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True)
                # Save
        writergif = animation.PillowWriter(fps=30)

        anim.save(outpath+'.gif',writer=writergif, savefig_kwargs={'facecolor':'white'})
        plt.show()
        fig.savefig('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/gradient_net.png', dpi=300)


# In[ ]:


plt.rcParams["font.family"] = "Arial"

for i,j in zip([main_grads, replace_grads, suppress_grads, clear_grads], 
               ['main', 'replace', 'suppress', 'clear']):
    
    arry = np.array(i.iloc[:, 1:])

    x = arry[:, 0]
    y = arry[:, 1]
    z = arry[:, 2]

    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    fig = plt.figure(figsize=(9.5, 5.5))

    ax1 = fig.add_subplot(121)
    ax1.set_facecolor((239/255, 239/255, 239/255))
    ax1.scatter(x, y, alpha=0.8, c=y, cmap='jet')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('Parcels Colored by Gradient', fontweight='bold', loc='left', fontname='Arial')
    ax1.set_xlabel('Gradient 1', fontweight='bold', fontname='Arial')
    ax1.set_ylabel('Gradient 2', fontweight='bold', fontname='Arial')
    ax1.set_xticks([min(x), max(x)])
    ax1.set_yticks([min(y), max(y)])

    ax2 = fig.add_subplot(122)
    ax2.set_facecolor((239/255, 239/255, 239/255))
    ax2.scatter(x, y, alpha=0.8, c=colors, cmap='jet')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('Parcels Colored by Network', fontweight='bold', loc='left', fontname='Arial')
    ax2.set_xlabel('Gradient 1', fontweight='bold', fontname='Arial')
    ax2.set_ylabel('Gradient 2', fontweight='bold', fontname='Arial')
    ax2.set_xticks([min(x), max(x)])
    ax2.set_yticks([min(y), max(y)])

    fig.tight_layout()

    #fig.savefig(f'/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/gradient_axes/{j}_grad_axes.png', dpi=700)


# In[ ]:


outpath= '/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/network_dispersion/group/'

#for i,j in zip([main_grads, replace_grads, suppress_grads, clear_grads],
#               ['main_networks','replace_networks','suppress_networks','clear_networks']): 
#               grad3d(i, outpath = outpath+j)
    
#for i,j in zip([main_grads, replace_grads, suppress_grads, clear_grads],
#               ['main_grads','replace_grads','suppress_grads','clear_grds']): 
#               grad3d(i, gradient=True, outpath = outpath+j)
            
for i,j in zip([main_grads, replace_grads, suppress_grads, clear_grads],
               ['main_networks_disp','replace_networks_disp','suppress_networks_disp','clear_networks_disp']): 
               network_dispersion(i ,outpath+j)


# In[ ]:


from glob import glob 

g1_path = '/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/subj/*_sm_vector/*_sm_vector_g1_all_ops.csv'
g2_path = '/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/subj/*_sm_vector/*_sm_vector_g2_all_ops.csv'
g3_path = '/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/subj/*_sm_vector/*_sm_vector_g3_all_ops.csv'

def sub_grads(paths, grad):
    g_dfs=[]
    for i in sorted(sorted(glob(paths, recursive = True))): 
        g_dfs.append((pd.read_csv(i).iloc[:, 2:]))
    sub_g = pd.concat(g_dfs).assign(gradient=grad)
    sub_g['sub'] = sub_g['sub'].str.replace('_sm_vector','').str.replace('sub','').astype(int)
    
    return sub_g

sub_g_list=[]
for i, j in zip([g1_path, g2_path, g3_path], range(1,4)):
    sub_g_list.append(sub_grads(i, j))

sub_gs=pd.concat(sub_g_list)

sub1= sub_gs.query('sub == 1') #high 
sub12 = sub_gs.query('sub == 12') #low


# In[ ]:


def long_grads(data, op):
    
    grad = (data[[op, 'gradient']]
     .reset_index()
     .pivot(index='index', columns='gradient', values=op)
     .reset_index(drop=True))
    
    grad = pd.DataFrame(np.array(grad)).reset_index()
    
    return grad
    
sub1_main = long_grads(sub1, 'maintain')
sub1_replace = long_grads(sub1, 'replace')
sub1_suppress = long_grads(sub1, 'suppress')
sub1_clear = long_grads(sub1, 'clear')

sub12_main = long_grads(sub12, 'maintain')
sub12_replace = long_grads(sub12, 'replace')
sub12_suppress = long_grads(sub12, 'suppress')
sub12_clear = long_grads(sub12, 'clear')


# In[ ]:


for i,j in zip([sub1_main, sub1_replace, sub1_suppress, sub1_clear], 
               ['main', 'replace', 'suppress', 'clear']):
    
    arry = np.array(i.iloc[:, 1:])

    x = arry[:, 0]
    y = arry[:, 1]
    z = arry[:, 2]

    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    fig = plt.figure(figsize=(9.5, 5.5))

    ax1 = fig.add_subplot(121)
    ax1.set_facecolor((239/255, 239/255, 239/255))
    ax1.scatter(x, y, alpha=0.8, c=y, cmap='jet')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('Parcels Colored by Gradient', fontweight='bold', loc='left', fontname='Arial')
    ax1.set_xlabel('Gradient 1', fontweight='bold', fontname='Arial')
    ax1.set_ylabel('Gradient 2', fontweight='bold', fontname='Arial')
    ax1.set_xticks([min(x), max(x)])
    ax1.set_yticks([min(y), max(y)])

    ax2 = fig.add_subplot(122)
    ax2.set_facecolor((239/255, 239/255, 239/255))
    ax2.scatter(x, y, alpha=0.8, c=colors, cmap='jet')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('Parcels Colored by Network', fontweight='bold', loc='left', fontname='Arial')
    ax2.set_xlabel('Gradient 1', fontweight='bold', fontname='Arial')
    ax2.set_ylabel('Gradient 2', fontweight='bold', fontname='Arial')
    ax2.set_xticks([min(x), max(x)])
    ax2.set_yticks([min(y), max(y)])

    fig.tight_layout()

    #fig.savefig(f'/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/gradient_axes/sub/sub1_{j}_grad_axes.png', dpi=700)


# In[ ]:


outpath= '/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/network_dispersion/subj/sub1/'

for i,j in zip([sub1_main,sub1_replace,sub1_suppress,sub1_clear],
               ['main_networks','replace_networks','suppress_networks','clear_networks']): 
               grad3d(i, outpath+j)
    
for i,j in zip([sub1_main,sub1_replace,sub1_suppress,sub1_clear],
       ['main_networks_disp','replace_networks_disp','suppress_networks_disp','clear_networks_disp']): 
               network_dispersion(i,outpath+j)


# In[ ]:


outpath= '/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/network_dispersion/subj/sub12/'

for i,j in zip([sub12_main,sub12_replace,sub12_suppress,sub12_clear],
               ['main_networks','replace_networks','suppress_networks','clear_networks']): 
               grad3d(i, outpath+j)
    
for i,j in zip([sub12_main,sub12_replace,sub12_suppress,sub12_clear],
               ['main_networks_disp','replace_networks_disp','suppress_networks_disp','clear_networks_disp']): 
               network_dispersion(i,outpath+j)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script gradient_network_plots.ipynb')


# In[ ]:


def mah_cent_dist(data, sub, op):
    
    data_sub = (data
                .query('sub =='+str(sub))[['sub', 'gradient', op]].reset_index()
                .pivot(index=['index', 'sub'], columns='gradient', values=op)
               ).reset_index()
   
    from scipy.spatial.distance import mahalanobis

    # Generate a random data frame with 95 rows and 3 columns
    df = data_sub.iloc[:, 2:]

    # Calculate the mean vector for each column
    mu = df.mean()

    cov = df.cov()
    # Calculate the inverse of the covariance matrix
    inv_cov = np.linalg.inv(cov)

    # Calculate the pairwise Mahalanobis distance between each row and the mean vector
    distances = []
    for index, row in df.iterrows():
        distance = mahalanobis(row, mu, inv_cov)
        distances.append(distance)

    # Convert the distances list to a pandas series and add it as a new column to the data frame
    df['md'] = pd.DataFrame(distances, index=df.index)
    df = df.assign(op = op, sub=sub, parcel=data_sub['index'])
    df = df.assign(cent_mah_md = df['op'] +'_'+ df['parcel'].astype(str) + '_md')
    df = df[['cent_mah_md', 'sub', 'op', 'md']]
    return df


def mah_dist(data, sub, op, parcels, network):
    
    data_sub = (data
                .query('sub =='+str(sub))[['sub', 'gradient', op]].reset_index()
                .pivot(index=['index', 'sub'], columns='gradient', values=op)
               ).reset_index().iloc[parcels]
   
    from scipy.spatial.distance import mahalanobis

    # Generate a random data frame with 95 rows and 3 columns
    df = data_sub.iloc[:, 2:]

    # Calculate the mean vector for each column
    mu = df.mean()

    cov = df.cov()
    # Calculate the inverse of the covariance matrix
    inv_cov = np.linalg.inv(cov)

    # Calculate the pairwise Mahalanobis distance between each row and the mean vector
    distances = []
    for index, row in df.iterrows():
        distance = mahalanobis(row, mu, inv_cov)
        distances.append(distance)

    # Convert the distances list to a pandas series and add it as a new column to the data frame
    df['md'] = pd.DataFrame(distances, index=df.index)
    df['md'] = df['md']*-1
    df = df.assign(op = op, sub=sub, parcel=data_sub['index'], net=network)
    df = df.assign(net_mah_md = df['op'] +'_'+ df['parcel'].astype(str) + '_md_'+df['net'])
    df = df[['net_mah_md', 'parcel', 'sub', 'op', 'md']]
    return df

