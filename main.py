import torch
from src.dataset import IMDBBertDataset
from src.model import BERTModel
from src.trainer import BertTrainer 
import random 
import numpy as np 
from tqdm import tqdm
import wandb

seed = 0
torch.manual_seed(seed) 
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

EMB_SIZE = 32
N_LAYERS = 2
N_HEADS = 8
DENSE_DIM = 64
DROPOUT = 0.5
N_EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

if __name__ == '__main__':
    print('preparing dataset. . .')
    wandb.init()
    # Since the initialization already calls the `prepare_dataset` function
    ## DATASET
    imdb_data = IMDBBertDataset('data/IMDB_Dataset.csv', ds_from=0, ds_to=20)
    ## MODEL
    bert_model = BERTModel(len(imdb_data.vocab), EMB_SIZE, N_LAYERS, N_HEADS, DENSE_DIM, DROPOUT).to(device)
    ## TRAIN
    bert_trainer = BertTrainer(bert_model, imdb_data, BATCH_SIZE, LEARNING_RATE)
    for epoch in tqdm(range(N_EPOCHS)):
        # IMPORTANT Q: there are nan but don't know where (found them once by luck in the inp_masked_sent)
        mlm_loss, nsp_loss = bert_trainer.train_epoch()
        wandb.log({"MLM Loss":mlm_loss, "NSP Loss":nsp_loss})
        print(f'For epoch {epoch}, the MLM training loss is: {mlm_loss}, while NSP classification loss is: {nsp_loss}')