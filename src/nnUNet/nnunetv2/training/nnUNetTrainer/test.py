from vesuvius_resource.losses.betti_losses import *
import torch
import torch.nn as nn

if __name__ == '__main__':
    loss_fn = FastMulticlassBettiMatchingLoss()