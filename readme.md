# CatDRX: Reaction-Conditioned Generative Model for Catalyst Design and Optimization
CatDRX, a catalyst discovery framework powered by a reaction-conditioned variational autoencoder generative model for generating catalysts and predicting their catalytic performance.

![graphical abstract](https://github.com/ohuelab/CatDRX/blob/main/blob/graphicalabstract.png?raw=true)

## Usage 💻

### Install environment 

This code was tested in Python 3.8 with PyTorch and rdkit 
- Using [Conda](https://www.anaconda.com/):
`conda create -f catdrx.yaml`
- Then, activate the environment
`conda activate catdrx`

### Dataset

- Prepare dataset in `dataset/` folder. Dataset should be in `.csv` format
- Indicate dataset metadata in `dataset/_dataset.py`
- The dataset metadata should include:
  - `file`: name of the dataset
  - `smiles`: column names for `reactant`, `reagent`, `product`, and `catalyst`. All columns are required, except for `reagent` (can be None)
  - `task`: column name for the task
  - `ids`: column name for the unique id
  - `splitting`: column name for the splitting (train, valid, test). If random splitting is used, the column name can be None
  - `predictiontask`: task name for the prediction task (yield, others)
  - `time`: column name for the reaction time
  - `condition_dict`: dictionary for the condition columns. For current version, only catalyst molecular weight is supported. Please refer to the example in the file.

### Pre-trainning

#### Pre-train your own model
1. Prepare the dataset
- Put the pre-training dataset in `dataset/` folder
- Insert dataset metadata in `dataset/_dataset.py`
- Example dataset: `dataset/ord.csv`

2. Pre-train the model
- Run the following command
```bash
python3 main_prediction.py \
--file [dataset] \
--epochs [epochs] \
--class_weight disabled \
--augmentation 5 \
--teacher_forcing
```

- [dataset] = name of dataset without `.csv` extension
- [epochs] = number of epochs
- [batch_size] = batch size
- For other parameters, please refer to `catcvae/setup.py` file

#### Use pre-trained model
- Download the pre-trained model from [available soon...]()

### Fine-tuning

1. Prepare the dataset
- Put the fine-tuning dataset in `dataset/` folder
- Insert dataset metadata in `dataset/_dataset.py`
- Example dataset: `dataset/sm.csv`

2. Fine-tuning the model 

#### For yield prediction task
- Run the following command
```bash
python3 main_finetune.py \
--file [dataset] \
--alpha [alpha] \
--beta [beta] \
--batch_size [batch_size] \
--epochs [epochs] \
--lr [lr] \
--class_weight [class_weight] \
--teacher_forcing \
--pretrained_file [pretrained_dataset] \
--pretrained_time [pretrained_dataset_folder]
```

#### For other catalystic activity prediction tasks
- Run the following command
```bash
python3 main_finetune_task.py \
--file [dataset] \
--alpha [alpha] \
--beta [beta] \
--batch_size [batch_size] \
--epochs [epochs] \
--lr [lr] \
--class_weight [class_weight] \
--teacher_forcing \
--pretrained_file [pretrained_dataset] \
--pretrained_time [pretrained_dataset_folder]
```
- [dataset] = name of dataset without `.csv` extension
- [alpha] = alpha value for the reconstruction loss function
- [beta] = beta value for the KL loss function
- [batch_size] = batch size
- [epochs] = number of epochs
- [lr] = learning rate
- [class_weight] = class weight for the loss function (disabled or enabled)
- [pretrained_dataset] = name of the pre-trained dataset without `.csv` extension
- [pretrained_dataset_folder] = name of the pre-trained dataset sub-folder (without `output_[seed]`)
- For other parameters, please refer to `catcvae/setup.py` file (Note: the core architecture parameters must be the same as the pre-trained model)
- The fined-tuned model will be saved in `dataset/[dataset]/output_[seed]_[dateandtime]` folder
- The performance results will be recorded in dataset folder with the file name `dataset/[dataset]/hyper_test.txt`

### Embedding space

1. Visualize embedding space
- Run the following command
```bash
python3 embeddingspace.py \
--file [dataset] \
--pretrained_file [finetuned_dataset] \
--pretrained_time [finetuned_dataset_folder]
```
- [dataset] = name of fine-tuned dataset without `.csv` extension
- [pretrained_dataset] = name of the fine-tuned dataset without `.csv` extension (mostly save as above)
- [pretrained_dataset_folder] = name of the fine-tuned dataset sub-folder (without `output_[seed]`)
- The fined-tuned model will be saved in `dataset/[pretrained_dataset]/output_[seed]_[pretrained_dataset_folder]` folder

### Generation and Optimization

1. Generate new catalysts
- Run the following command
```bash
python3 generation.py \
--file [dataset] \
--pretrained_file [finetuned_dataset] \
--pretrained_time [finetuned_dataset_folder] \
--correction [correction] \
--from_around_mol [from_around_mol] \
--from_around_mol_cond [from_around_mol_cond] \
--from_training_space [from_training_space] 
```
- [dataset] = name of dataset without `.csv` extension
- [pretrained_dataset] = name of the fine-tuned dataset without `.csv` extension
- [pretrained_dataset_folder] = name of the fine-tuned dataset sub-folder (without `output_[seed]`)
- [correction] = correction in post-processing step (disabled or enabled)
- [from_around_mol] = generate using sampled molecule from training set (disabled or enabled)
- [from_around_mol_cond] = generate using sampled molecule's condition (disabled or enabled)
- [from_training_space] = generate limited from the training space (disabled or enabled)
- For other setups related to number of molecules, task-specific validity, and generation parameters, please directly edit the `generation.py` file
- The fined-tuned model will be saved in `dataset/[pretrained_dataset]/output_[seed]_[pretrained_dataset_folder]` folder

2. Generate with optimization
- Run the following command
```bash
python3 optimization.py \
--file [dataset] \
--pretrained_file [finetuned_dataset] \
--pretrained_time [finetuned_dataset_folder] \
--opt_strategy [opt_strategy] \
```
- [dataset] = name of dataset without `.csv` extension
- [pretrained_dataset] = name of the fine-tuned dataset without `.csv` extension
- [pretrained_dataset_folder] = name of the fine-tuned dataset sub-folder (without `output_[seed]`)
- [opt_strategy] = optimization strategy (at_random', 'around_target')
- For other setups related to number of molecules, objective function, and optimization parameters, please directly edit the `optimization.py` file
- The fined-tuned model will be saved in `dataset/[pretrained_dataset]/output_[seed]_[pretrained_dataset_folder]` folder

## Citation 📃
> To be announced...