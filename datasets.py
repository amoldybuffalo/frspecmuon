import torch
from torch.utils.data import Dataset

class RandomLanguageModelDataset(Dataset):
    def __init__(self, tokens, block_size, steps_per_epoch, is_random = True):
        self.tokens = tokens
        self.block_size = block_size
        self.is_random = is_random
       

        if not self.is_random:
            self.steps_per_epoch = min(steps_per_epoch, int(len(tokens/block_size)))
        else:
            self.steps_per_epoch = steps_per_epoch

        


    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, j):
        if self.is_random:

            i = torch.randint(
                self.block_size,
                len(self.tokens),
                ()
            ).item()
            
        else:
            i = (j+1) * self.block_size
    
        x = self.tokens[i-self.block_size:i]

        return x