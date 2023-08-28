# Brazilian name generator

This repository contains training scripts and models for the generation of Brazilian names. The dataset is a CSV file with over 60k names from <https://github.com/datasets-br/prenomes>, whose source is IBGE.

Names are converted into n-grams and a Transformer is trained to predict the next character, given a partial name.

The models came from:

- <https://github.com/karpathy/ng-video-lecture/> (base models)
- <https://github.com/karpathy/makemore/> (n-gram training strategy)
- <https://github.com/karpathy/nanogpt/> (parallelized and flash implementation of multi-head self-attention)

Some pretty fun names are generated, check the sample at [sample.txt](sample.txt).

## Usage

A conda environment is provided, which can be generated and activated with:\

```sh
conda env create
conda activate brnames
```

A single module does everything, its documentation can be accessed with:

```sh
python -m brnames -h
```

To train a default module, use:

```sh
python -m brnames
```

Batch size is found automatically by PyTorch Lightning using the power rule to fill GPU memory.

To train multiple models using a predefined hyperparameter sweep with Ray Tune, use the `--tune` flag, which will ignore most other flags related to configuring the model and training. When using Tune, make sure your computer has a static IP and stable connection or you can have connection issues midway, even if running locally. The module with try to connect to an existing cluster and will start one if none are found.

If you are logged into Weights & Biases, you can log to a project called `brnames` by using the `--wandb` flag. Ray Tune also logs to TensorBoard by default in the `~/ray_results` directory.

## Generating names

To generate names using a trained model, use:

```sh
python -m brnames --gen <path to checkpoint file> <number of names to generate>
```

This will generate names in a file called `sample.txt`.

Checkpoint files are saved inside `~/ray_results`. A full example of the script call could be:

```sh
python -m brnames --gen ~/ray_results/brnames_asha/train_single_7a274_00000_0_activation=relu,dropout=0.3000,lr=0.0003,n_embd=128,n_head=2,n_layer=6,weight_decay=0.0050_2023-02-24_03-35-49/checkpoints/epoch=164-val_loss=1.6643.ckpt 25
```

## Model performance

| activation | n_embd | n_head | n_layer | dropout | lr      | weight_decay | iters | Loss/Train | Loss/Val |
|------------|--------|--------|---------|---------|---------|--------------|-------|------------|----------|
| relu       | 128    | 2      | 6       | 0.3     | 3.5E-04 | 5E-03        | 330   | 1.596      | 1.665    |
| gelu       | 384    | 6      | 5       | 0.4     | 3.5E-04 | 1E-03        | 96    | 1.640      | 1.669    |
| gelu       | 128    | 4      | 5       | 0.3     | 6.5E-04 | 5E-03        | 333   | 1.616      | 1.671    |
| relu       | 128    | 2      | 5       | 0.1     | 6.5E-04 | 5E-03        | 152   | 1.541      | 1.674    |
| relu       | 512    | 4      | 5       | 0.3     | 2.0E-04 | 1E-03        | 130   | 1.579      | 1.674    |
| gelu       | 384    | 2      | 5       | 0.3     | 3.5E-04 | 5E-03        | 121   | 1.529      | 1.680    |
| relu       | 256    | 4      | 6       | 0.1     | 8.0E-04 | 5E-03        | 98    | 1.477      | 1.680    |
| gelu       | 512    | 2      | 3       | 0.1     | 3.5E-04 | 1E-03        | 64    | 1.611      | 1.694    |
| relu       | 384    | 3      | 2       | 0.3     | 5.0E-04 | 5E-03        | 16    | 1.893      | 1.835    |
| relu       | 256    | 4      | 3       | 0.4     | 3.5E-04 | 1E-03        | 16    | 1.921      | 1.875    |
| relu       | 512    | 2      | 4       | 0.25    | 5.0E-04 | 5E-03        | 4     | 2.468      | 2.103    |
| gelu       | 256    | 2      | 2       | 0.5     | 8.0E-04 | 1E-03        | 1     | 2.557      | 2.467    |
| relu       | 256    | 2      | 2       | 0.25    | 3.5E-04 | 5E-03        | 1     | 2.537      | 2.471    |
| relu       | 128    | 4      | 3       | 0.25    | 5.0E-04 | 5E-03        | 1     | 2.648      | 2.607    |
| relu       | 128    | 4      | 4       | 0.5     | 8.0E-04 | 5E-03        | 1     | 2.647      | 2.614    |

All models trained with:

- AdamW + AMSGrad, beta1 = 0.9 and beta2 = 0.999
- ReduceLRonPlateau with 10 epochs of patience and scaling factor of 0.2.
- Early stopping with 20 epochs of patience.
- Vocabulary size = 27 (alphabet + start/end token) and block size = 15 (size of the largest names in the dataset).

## Name samples

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
