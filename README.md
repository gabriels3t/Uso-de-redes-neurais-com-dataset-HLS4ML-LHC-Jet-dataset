# Informação dos Dados
Os dados foram coletados no site :  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3601436.svg)](https://doi.org/10.5281/zenodo.3601436)

O dataset é configurado entre 3 modos:
<ol>
  <li>Lista de 16 HLFs (Jatos relacionados em alto nível de features), Tendo informações sobre : $\Sigma zlog(z)$, $C^{0}_{1}$ entre outros</li>
<li>Imagens de 100 $\times$ 100 pixel, representando jatos de partículas onde o eixo x é a pseudo-rapidez ($\eta$) e o eixo y o angulo azimutal <br>
  ($\phi$), que pode ser representado grosseiramente como o calorímetro eletromagnético utilizado nos experimentos do LHC, onde cade cada imagem é representada com os 150 partículas com maior energia </li>
<li>Uma lista constituinte de 30 partículas representadas por 16 features, sendo algumas delas : o momento nos eixos (x,y,z),$pT$,$\eta$ ( para obter a informação de todas as features , olhar o pdf "plot_treino_30Particulas.pdf")</li>
</ol>

# Instalando as Bibliotecas  
Antes de executar os scripts abaixo é preciso ter as mesmas bibliotecas pythons instaladas, é recomendado utilizar seguinte comando :
- Crie um ambiente virtual pelo conda:
  <code>conda env create -f environment.yml </code> <br/>
Pronto !

# Obtendo os Dados para o Treinamento do Modelo 
Foi utilizado os dados dos itens 2 e 3 .<br>
Por se tratar de vários arquivos  em .h5, foi criado scripts em Python como forma de tratar os dados e rearranjar todos os .h5 para apenas um arquivo .pt, que é um binário em Pytorch que facilita na economia de memória tanto em disco, quanto em memória ram.
<br>

## Baixando os Arquivos
<ol>
<li> Acessar o <a href="https://zenodo.org/records/3601436" target="_blank">link </a> </li>
<li> Fazer o Download dos seguintes arquivos: "hls4ml_LHCjet_30p_train.tar.gz", e "hls4ml_LHCjet_30p_val.tar.gz"
</li>
<li> Adicione os arquivos descompactados na pasta “data” (caso não tenha baixado a pasta do repositório, crie uma com o mesmo nome) </li>
</ol>

## Tratamento para o Dataset com Imagens 
Antes de executar o script "tratamento_dados_img.py", é preciso baixar o script “util.py”.  
Agora basta executar <br>
<code> python3 tratamento_dados_img.py </code> <br>
Apos a execução os dados vão ser retornados para a pasta "data".

## Tratamento para o Dataset das 30 Partículas 
Para o uso do Dataset com 30 partículas é mais simples, baixe os dados e coloque na pasta "data",apos isso baixe o script "util.py", pronto! agora basta executar:<br>
<code> python3 tratamento_dados_30particulas.py </code> <br>
Apos a execução os dados vão ser retornados para a pasta "data".

## Observação:
Se ocorrer um erro de "kill" ao tentar executar os scripts acima, é aconselhável reduzir o número de arquivos sendo processados. Para fazer isso, basta abrir o arquivo "tratamento_dados_30particulas.py" ou "tratamento_dados_img.py" e diminuir o valor da variável "quantidade_arquivos".
# Utilização dos modelos apresentados
