
import torch
import argparse
import numpy as np
from helper import *
from config.GlobalVariables import *
from SynthesisNetwork import SynthesisNetwork
from DataLoader import DataLoader
import style

L = 256


def main(params):
    np.random.seed(0)
    torch.manual_seed(0)

    device = 'cpu'

    net = SynthesisNetwork(weight_dim=256, num_layers=3).to(device)

    if not torch.cuda.is_available():
        net.load_state_dict(torch.load('./model/250000.pt', map_location=torch.device('cpu')))
        # net.load_state_dict(torch.load('./model_new/250000.pt', map_location=torch.device('cpu'))["model_state_dict"])

    dl = DataLoader(num_writer=1, num_samples=10, divider=5.0, datadir='./data/writers')

    all_loaded_data = []

    for writer_id in params.writer_ids:
        loaded_data = dl.next_batch(TYPE='TRAIN', uid=writer_id, tids=[i for i in range(params.num_samples)])
        all_loaded_data.append(loaded_data)


    if params.output == "image" and params.interpolate == "writer":
        if len(params.blend_weights) != len(params.writer_ids):
            raise ValueError("writer_ids must be same length as writer_weights")
        im = style.sample_blended_writers(params.blend_weights, params.target_word, net, all_loaded_data, device)
        im.convert("RGB").save(f'results/blend_{"+".join([str(i) for i in params.writer_ids])}.png')
    elif params.output == "grid" and params.interpolate == "character":
        im = style.sample_character_grid(params.grid_chars, params.grid_size, net, all_loaded_data, device)
        im.convert("RGB").save(f'results/grid_{"+".join(params.grid_chars)}.png')
    elif params.output == "video" and params.interpolate == "writer":
        style.writer_interpolation_video(params.target_word, params.frames_per_step, net, all_loaded_data, device)
    elif params.output == "video" and params.interpolate == "character":
        style.char_interpolation_video(params.video_chars, params.frames_per_step, net, all_loaded_data, device)
    elif params.output == "mdn":
        style.mdn_sampling_video(params.target_word, params.num_mdn_samples, params.mdn_max_scalar, net, all_loaded_data, device)
    else:
        print("Invalid task")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for generating samples with the handwriting synthesis model.')

    # parser.add_argument('--writer_id', type=int, default=80)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--generating_default', type=int, default=0)

    parser.add_argument('--output', type=str, default="image", choices=["image", "grid", "video", "mdn"])
    parser.add_argument('--interpolate', type=str, default="writer", choices=["writer", "character"])

    # IF OUTPUT IS MDN SAMPLING (NO INTERPOLATION)
    parser.add_argument('--mdn_max_scalar', type=float, default=2.5) 
    parser.add_argument('--num_mdn_samples', type=int, default=10)

    # IF INTERPOLATION

    # PARAMS FOR BOTH WRITER AND CHARACTER INTERPOLATION:
        # IF_BLEND
    parser.add_argument('--blend_weights', type=float, nargs="+", default=[0.5, 0.5])
        # IF GRID
    parser.add_argument('--grid_size', type=int, default=10)
        # IF VIDEO
    parser.add_argument('--frames_per_step', type=int, default=10)

    # PARAMS IF WRITER INTERPOLATION:
    parser.add_argument('--target_word', type=str, default="hello world")
    parser.add_argument('--writer_ids', type=int, nargs="+", default=[80, 120])
    
    # PARAMS IF CHARACTER INTERPOLATION:
        # IF BLEND
    parser.add_argument('--blend_chars', type=str, nargs="+", default = ["u", "g"])
        # IF GRID
    parser.add_argument('--grid_chars', type=str, nargs="+", default= ["x", "b", "u", "n"])
        # IF VIDEO
    parser.add_argument('--video_chars', type=str, default="abcdefghijklmnopqrstuvwxyz")

    main(parser.parse_args())
