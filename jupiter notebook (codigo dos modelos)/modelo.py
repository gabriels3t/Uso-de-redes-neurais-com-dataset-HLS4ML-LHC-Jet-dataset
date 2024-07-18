import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import optuna
from optuna.trial import TrialState

treino = torch.load("../data/treino_particle.pt")
teste = torch.load("../data/teste_particle.pt")

class CustomDataset(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data = data_tensor
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Obtendo as features e a label (target) da linha idx do tensor
        image = self.data[idx, 0:-1]
        label = self.data[idx, -1].long()

        # Aplicando transformações, se necessário
        if self.transform:
            image = self.transform(image)

        return image, label

dataset_treino = CustomDataset(treino)
dataset_teste = CustomDataset(teste)

batch_size = 128  
train_dataloader = DataLoader(dataset_treino, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset_teste, batch_size=batch_size, shuffle=True)

def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 2, 5)
    layers = []

    in_features = 16
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 10, 1200)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, 1))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)
entrada = 16 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Esta rodando em :{device}')


def objective(trial):
    # Define os hiperparâmetros que serão otimizados pelo Optuna
    entrada = 16
    
    # Define o modelo com os parâmetros sugeridos
    model = define_model(trial).to(device)

    # Define a função de custo e otimizador
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=trial.suggest_float('lr', 1e-5, 1e-1, log=True))

    epochs = trial.suggest_int("epochs", 150, 400)
    # Treina o modelo
    for epoch in range(epochs):
        model.train() # praparado para ser treinado
        erro_acumulativo = 0.0
        for _, (data, target) in enumerate(train_dataloader):
            target, datas =  target.float().to(device), data.float().to(device)
            pred = model(datas)
            perda = loss_function(pred,target.unsqueeze(1).to(device))

            optimizer.zero_grad() # zero os gradientes acumulados
            perda.backward() #Calculo da gradiente
            optimizer.step() # anda para a direção de menos erro

    # Avalia o modelo
    model.eval()  
    total_acertos = 0

    with torch.no_grad():  
        for _, (data, target) in enumerate(test_dataloader):
            target, datas = target.float().to(device), data.float().to(device)

            pred = model(datas)               
            pred = pred.argmax(dim=1, keepdim=True)
            total_acertos += pred.eq(target.view_as(pred)).sum().item()
            
    
    accuracy = total_acertos / len(test_dataloader)

    trial.report(accuracy, epoch)
    
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return accuracy

if __name__ == "__main__":

    # Configuração do estudo Optuna
    study = optuna.create_study(direction='maximize',storage='sqlite:///teste.db')
    study.optimize(objective, n_trials=100,timeout=7*24*60*60)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
# Print dos melhores hiperparâmetros encontrados
print('Melhores hiperparâmetros:')
print(study.best_params)

