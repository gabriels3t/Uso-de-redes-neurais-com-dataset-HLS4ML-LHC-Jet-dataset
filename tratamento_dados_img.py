import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import util
import gc

tqdm.monitor_interval = 0

def load_files(listOfFiles, file_name,maxFiles=-1):
    listParticulas = np.array([])

    total = len(listOfFiles[:maxFiles])
    for _, ifile in tqdm(enumerate(listOfFiles[:maxFiles]), desc='Processing',total=total):
        with h5py.File(ifile, "r") as f:
            particulas = np.array(f.get(file_name))
            listParticulas = np.concatenate((listParticulas, particulas), axis=0) if listParticulas.size else particulas
    
    return listParticulas

def preprocess_particles(imgP, data):
    # Separando as partículas pela suas categorias 
    list_q = np.array(imgP[data[:,-6]==1])
    list_g = np.array(imgP[data[:,-5]==1])
    list_W = np.array(imgP[data[:,-4]==1])
    list_Z = np.array(imgP[data[:,-3]==1])
    list_t = np.array(imgP[data[:,-2]==1])
    

    return list_q, list_g, list_Z, list_t, list_W

def juntar_dados_imagem_alvo(p_img, boson_W):
    dfs_concatenados = []
    
    TOTAL = len(p_img)
    print(f"Categorias: {TOTAL}")
    for j in range(TOTAL):
        classe = 1 if p_img[j] is boson_W else 0
        lista = p_img[j]
        resultados = []
        LINHA = len(lista)
        print(LINHA)
        for i in range(LINHA):
            dados_array = lista[i].reshape(100, 100).flatten()
            df_temp = pd.DataFrame(dados_array).T
            df_temp["Classe"] = classe
            dfs_concatenados.append(df_temp)

    resultado_final = pd.concat(dfs_concatenados, ignore_index=True)
    return resultado_final

def proporcao_teste(x):
  return int((26*x)/60)

def executando(path,caminho_saida,maxFiles):
    name_train = load_files(path,'jets',maxFiles)
    print("jatos carregados")
    data = load_files(path,'jetImage',maxFiles)
    print("imagens carregadas")
    data_q, data_g,  data_Z, data_t,data_W = preprocess_particles(data, name_train)

    del name_train
    del data

    data_P = np.array([data_q, data_g,  data_Z, data_t,data_W])
    df= juntar_dados_imagem_alvo(data_P,data_W)
    util.salvar_tensor_csv(df,caminho_saida)
    
    # Liberar memória manualmente 
    
    del df
    del data_q
    del data_g
    del data_Z
    del data_t
    del data_W
    del data_P
    
    # Forçar a coleta de lixo
    gc.collect()
    print("Executado com sucesso !")



# Executando
quantidade_arquivos = 60
path_train, path_test = util.carregar_dados("data")
saida = "data/treino.pt"
executando(path_train,saida,quantidade_arquivos)
saida = "data/teste.pt"
executando(path_test,saida,proporcao_teste(quantidade_arquivos))