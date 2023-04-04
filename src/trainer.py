"""
Initialize and Train
"""

from src.model import BERTModel # Q: why couldn't we call it as `model` directly and needed to add `src.model`, is the path raccording to entry point or what?
# from src.dataset import IMDBBertDataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import time
import datetime
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertTrainer:
    def __init__(self, model, dataset, batch_size, learning_rate):
        self.model = model 
        self.dataset = dataset
        self.batch_size = batch_size

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, drop_last=True)

        self.nsp_criterion = nn.BCEWithLogitsLoss().to(device) # Q: do we have here to ignore padding index? why only in mlm
        self.mlm_criterion = nn.NLLLoss(ignore_index=0).to(device)  
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def train_epoch(self):

        for i, data in enumerate(self.data_loader): 
            inp_masked_sent, attention_pad_mask, token_mask, inp_orig_sent, nsp_target = data # shape: bsz, seq_len
            self.optimizer.zero_grad()
            ## Q: sometimes found 'nan' in the token_mask_prediction (are these logits?), what does this mean?
            token_mask_prediction, nsp_prediction = self.model(inp_masked_sent, attention_pad_mask)
            
            # expand for same batch_size
            tm = token_mask.unsqueeze(-1).expand_as(token_mask_prediction)
            token_mask_prediction = token_mask_prediction.masked_fill(tm, 0) # to ignore all not masked and focus loss only on masks and its prediction

            nsp_loss = self.nsp_criterion(nsp_prediction, nsp_target)
            # shape (token_mask...): bsz, seq_len, vocab_size;; shape (inp_orig_sent): bsz, seq_len
            mlm_loss = self.mlm_criterion(token_mask_prediction.transpose(1, 2), inp_orig_sent) # BUG: 

            total_loss = nsp_loss + mlm_loss
            
            # We make backward first
            total_loss.backward()
            self.optimizer.step() # Dummy Q: for backward, we calculate gradients and for optimizer we change weights or which is what?
        
        return mlm_loss, nsp_loss

    # For checkpoints; cpd as in the code
    def save_checkpoint(self, epoch, step, loss):
        if not self.checkpoint_dir:
            return

        prev = time.time()
        name = f"bert_epoch{epoch}_step{step}_{datetime.utcnow().timestamp():.0f}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, self.checkpoint_dir.joinpath(name))

        print()
        print('=' * self._splitter_size)
        print(f"Model saved as '{name}' for {time.time() - prev:.2f}s")
        print('=' * self._splitter_size)
        print()

    def load_checkpoint(self, path: Path):
        print('=' * self._splitter_size)
        print(f"Restoring model {path}")
        checkpoint = torch.load(path)
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model is restored.")
        print('=' * self._splitter_size)
