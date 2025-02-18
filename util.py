import os
import sys
import torch

def adicionando_lista(dirs, files):
    diretorio = dirs + "/"
    return [diretorio + file_name for file_name in files]

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

def salvar_tensor_csv(data, tensor_path):
    tensor_data = torch.tensor(data.values, dtype=torch.float32)
    torch.save(tensor_data, tensor_path)
    print(f"Tensor salvo em {tensor_path}")