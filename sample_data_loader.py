#!/usr/bin/env python
# coding: utf-8

# ### Authour
# Masahiro Yasuda 
"""
PyTorch==1.7.0
"""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
import argparse
#data
from utils.distributed_dataloader import DistributedDataLoader
from utils.datautils_office import LoadDataset, TestLoadDataset

class Train():
    def __init__(self, cfg):
        self.cfg = cfg
        self.distributed_dataloader = DistributedDataLoader(cfg)

    def cleanup(self):
        dist.destroy_process_group()

    def train(self, local_rank, distargs):   
        rank = distargs.node * distargs.ngpus + local_rank#global rank
        world_size = distargs.ngpus * distargs.nodesize
        dist.init_process_group(backend='nccl', init_method=distargs.disturl,
                                    world_size = world_size, rank = rank)
        trainloader, \
        testloader = self.distributed_dataloader.setup_dataloader(LoadDataset(self.cfg,True),
                                                                   TestLoadDataset(self.cfg),
                                                                   world_size, rank)

        with torch.cuda.device(local_rank):                 
            for epoch in tqdm(range(self.cfg['MAX_EPOCH'])):         
                for i, sample in tqdm(enumerate(trainloader)): 
                    vx = sample['video1']
                    ax = sample['audio1']
                    #prediction pseudo code: pred = model(vx, ax)
                    eventlabel = sample['elabel1']
                    #loss pseudo code: loss = criterion(pred, eventlabel)
                    dist.barrier()

                for i, sample in tqdm(enumerate(testloader)): #half overlap evaluation
                    vx = sample['video1']
                    ax = sample['audio1']
                    eventlabel = sample['elabel1']
                    timelabel = sample['tlabel1']
                    dist.barrier()

        self.cleanup()

def train(local_rank, distargs, train_cfg):
    trainer = Train(train_cfg)
    trainer.train(local_rank, distargs)

def main():
    from config import train_cfg    
    parser = argparse.ArgumentParser(description='PyTorch Distributed Training')
    parser.add_argument('--trainmode', default='default', type=str)
    parser.add_argument('--node', default=0, type=int)
    parser.add_argument('--nodesize', default=1,type=int)
    parser.add_argument('--ngpus', default=4, type=int)
    parser.add_argument('--disturl',default='tcp://129.60.2.58:12345', type=str)
    distargs = parser.parse_args()
    train_cfg['mode'] = 'office'
    train_cfg['trainmode'] = distargs.trainmode

    mp.spawn(train,
            nprocs=distargs.ngpus,
            args=(distargs, train_cfg))
    
if __name__=="__main__":
     main()
