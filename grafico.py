import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns; sns.set()
import util
from matplotlib.lines import Line2D


def load_files(listOfFiles, file_name,maxFiles=-1):
    listParticulas = np.array([])

    for _, ifile in tqdm(enumerate(listOfFiles[:maxFiles]), total=maxFiles):
        with h5py.File(ifile, "r") as f:
            particulas = np.array(f.get(file_name))
            listParticulas = np.concatenate((listParticulas, particulas), axis=0) if listParticulas.size else particulas
    
    return listParticulas

def load_target(path_target,maxFiles):
    data = np.array([])
    for _, ifile in tqdm(enumerate(listOfFiles[:maxFiles]), total=maxFiles):
        with h5py.File(ifile, "r") as f:
            target = np.array(f.get("jets"))
            data = np.concatenate((data, target), axis=0) if data.size else target
            print(data.shape)
    
    return data 

def preprocess_particles(listP, data):
    # Separando as partículas pela suas categorias 
    list_q = np.array(listP[data[:,-6]==1])
    list_g = np.array(listP[data[:,-5]==1])
    list_W = np.array(listP[data[:,-4]==1])
    list_Z = np.array(listP[data[:,-3]==1])
    list_t = np.array(listP[data[:,-2]==1])

    # Retirando partículas com energia = 0 
    data_q = list_q[np.array(list_q[:,:,3]) > 0.]
    data_g = list_g[np.array(list_g[:,:,3]) > 0.]
    data_W = list_W[np.array(list_W[:,:,3]) > 0.]
    data_Z = list_Z[np.array(list_Z[:,:,3]) > 0.]
    data_t = list_t[np.array(list_t[:,:,3]) > 0.]
    
    return data_q, data_g, data_W, data_Z, data_t

# Executando
path_train, path_test =util.carregar_dados("data")


listP_train=load_files(path_train,'jetConstituentList',-1)
data_train = load_files(path_train,'jets',-1)

data_q_train, data_g_train, data_W_train, data_Z_train, data_t_train = preprocess_particles(listP_train, data_train)

# Configurações para os eixo x do histograma 
minval = [-1000, -1000, -1000, 0,0, 0., 0,  -np.pi, -2,-1.5,-np.pi,-0.6, -1, 0, -1,-0.75]
maxval = [ 1000,  1000,  1000,3500, 1,  1200,  1, np.pi, 1.4,1.5,np.pi, 0.6, 1, 2,1,0.75]

featureNames = ['$p_{x}$ [Gev]', '$p_{y}$ [Gev]', '$p_{z}$ [Gev]', '$E$ [Gev]', 
                'Relative~$E$ [Gev]', '$p_{T}$ [GeV]', 'Relative $p_{T}$ [GeV]',
                 '$\eta$', 'Relative $\eta$', 'Rotated $\eta$', '$\phi$', 'Relative $\phi$',
                'Rotated $\phi$', '$\\Delta R$', r'cos $\theta$', r'Relative cos $\theta$']

# Deixando todos com o mesmo minimo de entrada

minEntries = min(data_q_train.shape[0],data_g_train.shape[0],data_W_train.shape[0],
                data_Z_train.shape[0],data_t_train.shape[0])
data_q = data_q_train[:minEntries,:]
data_g = data_g_train[:minEntries,:]
data_W = data_W_train[:minEntries,:]
data_Z = data_Z_train[:minEntries,:]
data_t = data_t_train[:minEntries,:]


fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 16))

for i, ax in enumerate(axes.flat):
    ax.hist(np.ndarray.flatten(data_q[:,i]), range=(minval[i], maxval[i]), bins=50, density=True, histtype='step', fill=False, linewidth=1.5)
    ax.hist(np.ndarray.flatten(data_g[:,i]), range=(minval[i], maxval[i]), bins=50, density=True, histtype='step', fill=False, linewidth=1.5)
    ax.hist(np.ndarray.flatten(data_W[:,i]), range=(minval[i], maxval[i]), bins=50, density=True, histtype='step', fill=False, linewidth=1.5)
    ax.hist(np.ndarray.flatten(data_Z[:,i]), range=(minval[i], maxval[i]), bins=50, density=True, histtype='step', fill=False, linewidth=1.5)
    ax.hist(np.ndarray.flatten(data_t[:,i]), range=(minval[i], maxval[i]), bins=50, density=True, histtype='step', fill=False, linewidth=1.5)
    
    ax.set_xlabel(featureNames[i], fontsize=14)
    ax.set_ylabel('Densidade de Probabilidade', fontsize=14)
    
    ax.semilogy()
labelCat = ["quark", "gluon", "W", "Z", "top"]
handles = [Line2D([], [], c=f'C{i}', label=labelCat[i]) for i in range(len(labelCat))]
fig.legend(handles=handles, loc="upper right", fontsize=14, frameon=False)
plt.tight_layout()
plt.draw()
plt.savefig('plot_treino.pdf' , dpi=250)
plt.close()

print("Finalizado")