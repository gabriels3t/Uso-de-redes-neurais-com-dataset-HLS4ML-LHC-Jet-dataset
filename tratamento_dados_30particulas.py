import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import util


def load_files(listOfFiles, file_name,maxFiles=-1):
    listParticulas = np.array([])

    for _, ifile in tqdm(enumerate(listOfFiles[:maxFiles]), total=maxFiles):
        with h5py.File(ifile, "r") as f:
            particulas = np.array(f.get(file_name))
            listParticulas = np.concatenate((listParticulas, particulas), axis=0) if listParticulas.size else particulas
    
    return listParticulas

    # Retirando partículas com energia = 0 
def filtrar_particulas_com_energia_zero(particles):
    mask = np.any(particles[:,:,3] > 0, axis=1)
    return particles[mask]

def preprocess_particles(listP, data):
    # Separando as partículas pela suas categorias 
    list_q = np.array(listP[data[:,-6]==1])
    list_g = np.array(listP[data[:,-5]==1])
    list_W = np.array(listP[data[:,-4]==1])
    list_Z = np.array(listP[data[:,-3]==1])
    list_t = np.array(listP[data[:,-2]==1])
    
    data_q = filtrar_particulas_com_energia_zero(list_q)
    data_g = filtrar_particulas_com_energia_zero(list_g)
    data_W = filtrar_particulas_com_energia_zero(list_W)
    data_Z = filtrar_particulas_com_energia_zero(list_Z)
    data_t = filtrar_particulas_com_energia_zero(list_t)

    return data_q, data_g, data_Z, data_t, data_W

def concatenando_dados(data, feature_names, boson_W):
    dfs_concatenados = []
    CATEGORIAS = len(data)
    print(f"Categorias: {CATEGORIAS}")

    for k in range(CATEGORIAS):

        classe = 1 if data[k] is boson_W else 0
        lista = data[k]
        TAMANHO_LISTA = len(lista)
        print(lista.shape)
        
        for i in range(TAMANHO_LISTA):
            dados_array = np.concatenate([lista[i][j].reshape(1, -1) for j in range(30)], axis=0)
            df_temp = pd.DataFrame(dados_array, columns=feature_names)
            df_temp['Classe'] = classe
            dfs_concatenados.append(df_temp)

    resultado_final = pd.concat(dfs_concatenados, ignore_index=True)
    return resultado_final

def feature_names():
    return ['$p_{x}$ [Gev]', '$p_{y}$ [Gev]', '$p_{z}$ [Gev]', '$E$ [Gev]', 
                'Relative~$E$ [Gev]', '$p_{T}$ [GeV]', 'Relative $p_{T}$ [GeV]',
                 '$\eta$', 'Relative $\eta$', 'Rotated $\eta$', '$\phi$', 'Relative $\phi$',
                'Rotated $\phi$', '$\\Delta R$', r'cos $\theta$', r'Relative cos $\theta$']

def executando(path,caminho_saida):
    label = feature_names()
    listP = load_files(path,'jetConstituentList',-1)
    data = load_files(path,'jets',-1)
    data_q, data_g,  data_Z, data_t,data_W = preprocess_particles(listP, data)
    data_P = np.array([data_q, data_g,  data_Z, data_t,data_W])
    df = concatenando_dados(data_P,label,data_W)
    util.salvar_tensor_csv(df,caminho_saida)
    print("Executado com sucesso !")          

# Executando
path_train, path_test = util.carregar_dados("data")

saida = "data/treino_particle.pt"
executando(path_train,saida)
saida = "data/teste_particle.pt"
executando(path_test,saida)
