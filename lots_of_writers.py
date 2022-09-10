import torch
import argparse
import numpy as np
from helper import *
from config.GlobalVariables import *
from SynthesisNetwork import SynthesisNetwork
from DataLoader import DataLoader
import convenience

device = 'cpu'
num_samples = 10

net = SynthesisNetwork(weight_dim=256, num_layers=3).to(device)

if not torch.cuda.is_available():
    try: # retrained model also contains loss in dict 
        net.load_state_dict(torch.load('./model/250000.pt', map_location=torch.device(device))["model_state_dict"])
    except:
        net.load_state_dict(torch.load('./model/250000.pt', map_location=torch.device(device)))
    
dl = DataLoader(num_writer=1, num_samples=10, divider=5.0, datadir='./data/writers')
for writer_id in range(0, 170):
    loaded_data = dl.next_batch(TYPE='TRAIN', uid=writer_id, tids=list(range(num_samples)))
    image = convenience.mdn_single_sample("hello world quick brown fox", 1, 0, net, [loaded_data], device).convert("RGB")
    image.save(f"results/all_writers/writer_{writer_id}.png")
