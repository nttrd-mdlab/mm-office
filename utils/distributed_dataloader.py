import torch
from torch.utils.data import DataLoader

class DistributedDataLoader():
    def __init__(self, cfg):
        self.cfg = cfg

    def setup_dataloader(self, trainset, testset, world_size, rank):                                                                
    #def setup_dataloader(self, trainset, world_size, rank):                                                                
        #Sampler        
        trainsampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)

        trainloader = DataLoader(trainset, batch_size=self.cfg["batch_num"],
                                sampler=trainsampler, pin_memory=True,num_workers=8)
        testloader = DataLoader(testset, batch_size=4, shuffle=False, pin_memory=True,num_workers=4)
        return trainloader, testloader