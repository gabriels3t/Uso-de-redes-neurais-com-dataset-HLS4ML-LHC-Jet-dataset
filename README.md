# Informação dos Dados
Os dados foram coletados no site :  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3601436.svg)](https://doi.org/10.5281/zenodo.3601436)

O dataset é configurado entre 3 modos:
<ol>
  <li>Lista de 16 HLFs (Jatos relacionados em alto nível de features), Tendo informações sobre : $\Sigma zlog(z)$, $C^{0}_{1}$ entre outros</li>
<li>Imagens de 100 $\times$ 100 pixel, representando jatos de partículas onde o eixo x é a pseudo-rapidez ($\eta$) e o eixo y o angulo azimutal <br>
  ($\phi$), que pode ser representado grosseiramente como o calorímetro eletromagnético utilizado nos experimentos do LHC, onde cade cada imagem é representada com os 150 partículas com maior energia </li>
<li>Uma lista constituinte de 30 partículas representadas por 16 features, sendo algumas delas : o momento nos eixos (x,y,z),$pT$,$\eta$ ( para obter a informação de todas as features , olhar o pdf "plot_treino_30Particulas.pdf")</li>
</ol>

# Obtendo os Dados para o Treinamento do Modelo 
Foi utilizado os dados de imagens e lista constituinte de 30 partículas.<br>
Por se tratar de vários arquivos  em .h5, foi criado scripts em Python como forma de tratar os dados e rearranjar todos os .h5 para apenas um arquivo .pt, que é um binário em Pytorch que facilita na economia de memória tanto em disco, quanto em memória ram.
<br>
### Instalando as Bibliotecas  
Antes de executar os scripts abaixo é preciso ter as mesmas bibliotecas pythons instaladas , com isso é recomendado utilizar seguintes comandos :
- Crie um ambiente virtual:
  <code>python3 -m venv nome_do_ambiente_virtual </code>
- Apos criar o ambiente virtual é preciso ser ativado:
  <code>source nome_do_ambiente_virtual/bin/activate </code>
  <br>
  caso seja em um ambiente windows:
  <code>nome_do_ambiente_virtual\Scripts\Activate </code>
- Apos estar dentro do ambiente, basta instalar as bibliotecas (caso tenha duvidas de quais instalar, basta dar o seguinte comando): <br> 
  <code>pip install -r requirements.txt </code>
## Tratamento para o Dataset com Imagens 
Antes de executar o script "tratamento_dados_img.py", é preciso baixar os dados do site, colocar em uma pasta chamada "data", e baixar os arquivos "target_test_raw.csv" e "target_train_raw.csv" do repositório ( disponível na pasta "data"), e o script util.py.  
Agora basta executar <br>
<code> python3 tratamento_dados_img.py </code> <br>
Apos a execução os dados vão ser retornados para a pasta "data".

## Tratamento para o Dataset das 30 Partículas 
Para o uso do Dataset com 30 partículas é mais simples, baixe os dados e coloque na pasta "data",apos isso baixe o script "util.py", pronto! agora basta executar:<br>
<code> python3 tratamento_dados_30particulas.py </code> <br>
<<<<<<< HEAD
<<<<<<< HEAD
Apos a execução os dados vão ser retornados para a pasta "data".
=======
Apos a execução os dados vão ser retornados para a pasta "data".
>>>>>>> 1dc3bb01cd67cd301eb1eebfcc77bd3392df2802
=======
Apos a execução os dados vão ser retornados para a pasta "data".
>>>>>>> 1dc3bb01cd67cd301eb1eebfcc77bd3392df2802
