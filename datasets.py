import torch
from torch.utils.data import Dataset

class RandomLanguageModelDataset(Dataset):
    def __init__(self, tokens, block_size, steps_per_epoch):
        self.tokens = tokens
        self.block_size = block_size
        self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, _):
        i = torch.randint(
            self.block_size,
            len(self.tokens),
            ()
        ).item()

        x = self.tokens[i-self.block_size:i]

        return x