
import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
def conv_layer(chanIn, chanOut, kernel_size = 3,strd=1, padding=1, drop=.04):
    return nn.Sequential(
        nn.Conv2d(chanIn, chanOut, kernel_size,stride=strd, padding=padding),        
        nn.BatchNorm2d(chanOut),
        nn.ReLU(),
        nn.Dropout(drop)
        )

def discr_dense(chanIn):
    return nn.Sequential(
        Flatten(),
        nn.Linear(chanIn, 1),
        nn.Sigmoid() #...switch to bcewlogits if you remove sigmoid
        )

def gen_dense(chanIn, chanOut):
    return nn.Sequential(
        nn.Linear(chanIn, chanOut),
        nn.ReLU()
        )
