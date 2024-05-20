import h5py
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch

def adicionando_lista(dirs, files):
    diretorio = dirs + "/"
    return [diretorio + file_name for file_name in files]

def loadfiles_np(listOfFiles, maxFiles=-1):
    listofDf = []
    for _, ifile in tqdm(enumerate(listOfFiles[:maxFiles]), total=maxFiles):
        with h5py.File(ifile, "r") as f:
            tmpDf = np.array(f.get("jetImageHCAL"))
            listofDf.append(tmpDf)
    return np.stack(listofDf)

def carregar_dados(path):
    paths = []
    cont = 0
    for dirs, _, files in os.walk(path):
        if cont == 1:
            print(f'Adicionando os dados de {dirs[-4:]} na lista ')
            paths.append(adicionando_lista(dirs, files))
        if cont == 2:
            print(f'Adicionando os dados de {dirs[-5:]} na lista ')
            paths.append(adicionando_lista(dirs, files))
        cont += 1
    return paths

def salvar_numpy_array(dados, caminho):
    with open(caminho, 'wb') as f:
        np.save(f, dados)

def juntar_dados_imagem_alvo(path_imagens, target_csv, caminho_saida):
    df_image = loadfiles_np(path_imagens, -1)
    target_train = pd.read_csv(target_csv)

    dfs_concatenados = []
    TOTAL = df_image.shape[0]

    for j in range(TOTAL):
        resultados = []
        LINHA = df_image[j].shape[0]

        for i in range(LINHA):
            resultados.append(df_image[j][i].astype('uint8').reshape(100, 100).flatten())

        resultado = np.array(resultados)
        treino = pd.DataFrame(resultado.reshape(LINHA,-1))

        treino['Classe'] = target_train["j_w"]
        target_train["j_w"] = target_train.iloc[LINHA:]
        target_train = target_train.dropna().reset_index(drop=True)
        dfs_concatenados.append(treino)

    resultado_final = pd.concat(dfs_concatenados)
    resultado_final.to_csv(caminho_saida)

    isna = (resultado_final.isna().sum().sum()/10001)
    print("Contagem de NaN em cada coluna:")
    print(isna)

    return resultado_final


def salvar_tensor_csv(csv_path, tensor_path, dtypes):
    data = pd.read_csv(csv_path, dtype=dtypes)
    tensor_data = torch.tensor(data.values, dtype=torch.float32)
    torch.save(tensor_data, tensor_path)
    print(f"{csv_path} convertido para tensor e salvo em {tensor_path}")




# Executando
path_train, path_test = carregar_dados("dados")
salvar_numpy_array(loadfiles_np(path_train, -1), 'dados/hcal_train.npy')
salvar_numpy_array(loadfiles_np(path_test, -1), 'dados/hcal_test.npy')

path_test = "dados/hcal_test.npy"
juntar_dados_imagem_alvo(path_test, "dados/target_test_raw.csv", 'dados/teste.csv')

path_treino= "dados/hcal_train.npy"
juntar_dados_imagem_alvo(path_treino, "dados/target_train_raw.csv", 'dados/treino.csv')

dtypes = {f'{i}': 'float32' for i in range(0, 10000)}
dtypes['Unnamed: 0'] = 'int64'
dtypes['Classe'] = 'int64'

salvar_tensor_csv("dados/treino.csv", "dados/treino.pt", dtypes)
salvar_tensor_csv("dados/teste.csv", "dados/teste.pt", dtypes)
