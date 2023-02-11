# Gerador de nome de brasileiro

Esse repositório implementa um script de treinamento para treinar um modelo que gera nome de brasileiro. **Teoricamente**, ele gera sequências de caracteres, dado um conjunto de sequências de caractere para treinamento, mas eu queria ver se conseguia gerar uns nomes engraçados, então aqui estamos...

O arquivo com os nomes veio de <https://github.com/datasets-br/prenomes>.

O modelo veio de <https://github.com/karpathy/ng-video-lecture/> e foi adaptado para gerar nomes ao invés de poesias de Shakespeare.

A estratégia de treinamento usando n-gramas veio de <https://github.com/karpathy/makemore/>

## Uso

Para criar um ambiente conda:

```sh
conda env create
conda activate brnames
```

A documentação de uso está em:

```sh
python -m brnames.train -h
```

O comando abaixo treina um modelo padrão:

```sh
python -m brnames.train
```

Foram implementadas opções para selecionar diferentes otimizadores e treinar o modelo com precisão mista automática (PyTorch AMP).

Pesos de modelo são salvos no diretório `weights/` e carregados antes do treino caso algum arquivo compatível exista.

O *logging* é executado no terminal e no TensorBoard, no diretório `runs/`. Na etapa de avaliação do modelo, um conjunto de nomes é disponibilizado em ambos os lugares para conferência.

## Gerando nomes

Para gerar uma lista de nomes, primeiro treine um modelo e depois execute o script de treino novamente com a flag `--gen`. Os pesos do modelo serão recuperados do diretório `weights/` e um arquivo chamado `sample.txt` será gerado com 1000 nomes.

## Amostras

```
jossenira
handreja
eklem
farociana
maeusinar
talhani
neurio
valtesio
gualiene
ledja
helberte
gilsineidi
djosy
zoira
paulenice
niziane
gersinilda
abert
erito
linaira
queta
malne
ederlanda
evertro
kayne
cleilson
lucivania
```
</details>
