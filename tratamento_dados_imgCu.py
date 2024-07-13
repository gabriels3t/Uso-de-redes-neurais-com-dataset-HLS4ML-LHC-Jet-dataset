import cupy as cp
import h5py
import pandas as pd
from tqdm import tqdm
import torch
import util
import gc

"""
def load_files(listOfFiles, file_name, maxFiles=-1):
    listParticulas = cp.array([], dtype=cp.float32)

    total = len(listOfFiles[:maxFiles])
    for _, ifile in tqdm(enumerate(listOfFiles[:maxFiles]), desc='Processing', total=total):
        with h5py.File(ifile, "r") as f:
            particulas = cp.array(f.get(file_name), dtype=cp.float32)
            listParticulas = cp.concatenate((listParticulas, particulas), axis=0) if listParticulas.size else particulas

    return listParticulas
"""
def load_files(listOfFiles, file_name, maxFiles=-1, chunk_size=50):
    batches = []

    total = len(listOfFiles[:maxFiles])
    for _, ifile in tqdm(enumerate(listOfFiles[:maxFiles]), desc='Processing', total=total):
        with h5py.File(ifile, "r") as f:
            particulas = cp.array(f.get(file_name), dtype=cp.float32)

            # Processar em pequenos lotes para reduzir alocação de memória
            for chunk_start in range(0, len(particulas), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(particulas))
                chunk = particulas[chunk_start:chunk_end]

                batches.append(chunk)

    return batches


def preprocess_particles(imgP, data):
    list_q = []
    list_g = []
    list_W = []
    list_Z = []
    list_t = []
    
    # Iterar sobre os dados fornecidos (que podem ser um gerador ou iterável)
    for batch in data:
        # Aplicar as condições de seleção em cada lote
        mask_q = batch[:, -6] == 1
        mask_g = batch[:, -5] == 1
        mask_W = batch[:, -4] == 1
        mask_Z = batch[:, -3] == 1
        mask_t = batch[:, -2] == 1
        
        # Selecionar e converter para cupy arrays (GPU arrays)
    
        list_q.append(cp.array(imgP[mask_q], dtype=cp.float32))
        list_g.append(cp.array(imgP[mask_g], dtype=cp.float32))
        list_W.append(cp.array(imgP[mask_W], dtype=cp.float32))
        list_Z.append(cp.array(imgP[mask_Z], dtype=cp.float32))
        list_t.append(cp.array(imgP[mask_t], dtype=cp.float32))
    
    # Converter listas para cupy arrays (GPU arrays), se disponível
    list_q = cp.concatenate(list_q, axis=0)
    list_g = cp.concatenate(list_g, axis=0)
    list_W = cp.concatenate(list_W, axis=0)
    list_Z = cp.concatenate(list_Z, axis=0)
    list_t = cp.concatenate(list_t, axis=0)

    return list_q, list_g, list_Z, list_t, list_W

def juntar_dados_imagem_alvo(p_img, boson_W):
    dfs_concatenados = []

    TOTAL = len(p_img)
    print(f"Categorias: {TOTAL}")
    for j in range(TOTAL):
        classe = 1 if cp.array_equal(p_img[j], boson_W) else 0
        lista = p_img[j]
        resultados = []
        LINHA = len(lista)
        print(LINHA)
        for i in tqdm(range(LINHA)):
            dados_array = lista[i].reshape(100, 100).flatten()
            df_temp = pd.DataFrame(cp.asnumpy(dados_array).T)
            df_temp["Classe"] = classe
            dfs_concatenados.append(df_temp)

    resultado_final = pd.concat(dfs_concatenados, ignore_index=True)
    return resultado_final

def proporcao_teste(x):
    return int((26 * x) / 60)

def executando(path, caminho_saida, maxFiles):
    name_train = load_files(path, 'jets', maxFiles)
    print("jatos carregados")
    data = load_files(path, 'jetImage', maxFiles)
    print("imagens carregadas")
    data_q, data_g, data_Z, data_t, data_W = preprocess_particles(data, name_train)

    del name_train
    del data

    data_P = cp.array([data_q, data_g, data_Z, data_t, data_W], dtype=object)
    df = juntar_dados_imagem_alvo(data_P, data_W)
    util.salvar_tensor_csv(df, caminho_saida)

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
saida = "data/teste.pt"
executando(path_test, saida, quantidade_arquivos)
saida = "data/treino.pt"
executando(path_train, saida, proporcao_teste(quantidade_arquivos))
