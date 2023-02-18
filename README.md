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

## Desempenho

| Modelo | Ativações  | Embedding | Cabeças de auto-atenção | Blocos de auto-atenção | Dropout | Inicialização  | Erro |
|--------|------------|-----------|-------------------------|------------------------|---------| ---------------|------|
| 1      | ReLU       | 128       | 4                       |                        | 0.2     | N(0; 0,02)     | 1.69 |
| 2      | ReLU       | 64        | 3                       | 2                      | 0.2     | N(0; 0,02)     | 1.74 |
| 3      | ReLU       | 128       | 4                       | 3                      | 0.2     | N(0; 0,02)     | 1.68 |
| 4      | ReLU       | 128       | 4                       | 3                      | 0.2     | N(0; 0,02)     | 1.67 |
| 5      | ReLU       | 384       | 3                       | 3                      | 0.2     | N(0; 0,02)     | 1.66 |
| 6      | TanH       | 384       | 3                       | 3                      | 0.2     | N(0; 0,02)     | 1.7  |
| 7      | ReLU       | 384       | 3                       | 3                      | 0.2     | Kaiming-normal | 1.69 |
| 8      | Leaky-ReLU | 384       | 3                       | 3                      | 0.2     | N(0; 0,02)     | 1.67 |

## Amostras

```
petralino
ivalmir
maerio
bosca
edjames
ellyda
vaelica
jessicleia
sylverio
zaqueu
heinrick
kaycke
carlena
valdeice
aguinailton
marailson
```
