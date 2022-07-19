
import os
import torch
import pickle
import argparse
import numpy as np
from helper import *
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from config.GlobalVariables import *
from tensorboardX import SummaryWriter
from SynthesisNetwork import SynthesisNetwork
from DataLoader import DataLoader


def main(params):
    np.random.seed(0)
    torch.manual_seed(0)

    device = 'cpu'

    net = SynthesisNetwork(weight_dim=256, num_layers=3).to(device)

    if not torch.cuda.is_available():
        # net.load_state_dict(torch.load('./model_original/250000.pt', map_location=torch.device('cpu')))
        net.load_state_dict(torch.load('./model/250000.pt', map_location=torch.device('cpu'))["model_state_dict"])

    dl = DataLoader(num_writer=1, num_samples=10, divider=5.0, datadir='./data/writers')

    writer_ids = [80, 120]
    writer_weights = [1, 0]
    all_loaded_data = []

    for writer_id in writer_ids:
        loaded_data = dl.next_batch(TYPE='TRAIN', uid=writer_id, tids=[i for i in range(params.num_samples)])
        all_loaded_data.append(loaded_data)

    target_word = "the quick brown fox"
    if len(target_word) > 0:
        im = sample(writer_weights, target_word, net, all_loaded_data, device)
        im.convert("RGB").save(f'results/writer_120/output.png')


def get_mean_global_W(net, loaded_data, device):
    """gets the mean global style vector for a given writer"""
    [_, _, _, _, _, _, all_word_level_stroke_in, all_word_level_stroke_out, all_word_level_stroke_length, all_word_level_term, all_word_level_char, all_word_level_char_length, all_segment_level_stroke_in, all_segment_level_stroke_out,
        all_segment_level_stroke_length, all_segment_level_term, all_segment_level_char, all_segment_level_char_length] = loaded_data

    batch_word_level_stroke_in = [torch.FloatTensor(a).to(device) for a in all_word_level_stroke_in]
    batch_word_level_stroke_out = [torch.FloatTensor(a).to(device) for a in all_word_level_stroke_out]
    batch_word_level_stroke_length = [torch.LongTensor(a).to(device).unsqueeze(-1) for a in all_word_level_stroke_length]
    batch_word_level_term = [torch.FloatTensor(a).to(device) for a in all_word_level_term]
    batch_word_level_char = [torch.LongTensor(a).to(device) for a in all_word_level_char]
    batch_word_level_char_length = [torch.LongTensor(a).to(device).unsqueeze(-1) for a in all_word_level_char_length]
    batch_segment_level_stroke_in = [[torch.FloatTensor(a).to(device) for a in b] for b in all_segment_level_stroke_in]
    batch_segment_level_stroke_out = [[torch.FloatTensor(a).to(device) for a in b] for b in all_segment_level_stroke_out]
    batch_segment_level_stroke_length = [[torch.LongTensor(a).to(device).unsqueeze(-1) for a in b] for b in all_segment_level_stroke_length]
    batch_segment_level_term = [[torch.FloatTensor(a).to(device) for a in b] for b in all_segment_level_term]
    batch_segment_level_char = [[torch.LongTensor(a).to(device) for a in b] for b in all_segment_level_char]
    batch_segment_level_char_length = [[torch.LongTensor(a).to(device).unsqueeze(-1) for a in b] for b in all_segment_level_char_length]

    with torch.no_grad():
        word_inf_state_out = net.inf_state_fc1(batch_word_level_stroke_out[0])
        word_inf_state_out = net.inf_state_relu(word_inf_state_out)
        word_inf_state_out, _ = net.inf_state_lstm(word_inf_state_out)

        user_word_level_char = batch_word_level_char[0]
        user_word_level_term = batch_word_level_term[0]

        original_Wc = []
        word_batch_id = 0

        curr_seq_len = batch_word_level_stroke_length[0][word_batch_id][0]
        curr_char_len = batch_word_level_char_length[0][word_batch_id][0]

        char_vector = torch.eye(len(CHARACTERS))[user_word_level_char[word_batch_id][:curr_char_len]].to(device)
        current_term = user_word_level_term[word_batch_id][:curr_seq_len].unsqueeze(-1)
        split_ids = torch.nonzero(current_term)[:, 0]

        char_vector_1 = net.char_vec_fc_1(char_vector)
        char_vector_1 = net.char_vec_relu_1(char_vector_1)

        char_out_1 = char_vector_1.unsqueeze(0)
        char_out_1, (c, h) = net.char_lstm_1(char_out_1)
        char_out_1 = char_out_1.squeeze(0)
        char_out_1 = net.char_vec_fc2_1(char_out_1)
        char_matrix_1 = char_out_1.view([-1, 1, 256, 256])
        char_matrix_1 = char_matrix_1.squeeze(1)
        char_matrix_inv_1 = torch.inverse(char_matrix_1)

        W_c_t = word_inf_state_out[word_batch_id][:curr_seq_len]
        W_c = torch.stack([W_c_t[i] for i in split_ids])
        original_Wc.append(W_c)

        W = torch.bmm(char_matrix_inv_1, W_c.unsqueeze(2)).squeeze(-1)
        mean_global_W = torch.mean(W, 0)
        return mean_global_W


def get_writer_blend_W_c(writer_weights, all_writer_Ws, all_writer_Cs):
    """
    generates character-dependent style-dependent DSDs for each character/segement in target_word

    n is the number of writers
    M is the number of characters in the target word
    L is the latent vector size (in this case 256)

    input:
    writer_weights, a list of length n weights for each writer that sum to one
    all_writer_Ws, an n x M x L tensor representing each weiter's style vector for every character
    all_writer_Cs, an n x M x L x L tensor representing the style's correspodning character matrix

    output:
    an M x 1 x L
    """
    n, M, L = all_writer_Ws.shape
    weights_tensor = torch.tensor(writer_weights).repeat_interleave(M * L).reshape(n, M, L)  # repeat accross remaining dimensions
    W_vectors = (weights_tensor * all_writer_Ws).sum(axis=0)  # take weighted sum accross writers axis
    W_vectors = W_vectors.unsqueeze(-1)  # prepare for batch multiplication
    char_matrices = all_writer_Cs[0, :, :, :]  # matrices are independent of writer

    return torch.bmm(char_matrices, W_vectors).reshape(M, 1, L)


def get_DSD(net, target_word, writer_mean_Ws, all_loaded_data, device):
    """returns a style vector and character matrix for each character/segment in target_word"""

    n = len(all_loaded_data)
    M = len(target_word)
    L = 256
    all_writer_Ws = torch.zeros(n, M, L)
    all_writer_Cs = torch.zeros(n, M, L, L)

    for i in range(n):
        np.random.seed(0)

        [_, _, _, _, _, _, all_word_level_stroke_in, all_word_level_stroke_out, all_word_level_stroke_length, all_word_level_term, all_word_level_char, all_word_level_char_length, all_segment_level_stroke_in, all_segment_level_stroke_out,
            all_segment_level_stroke_length, all_segment_level_term, all_segment_level_char, all_segment_level_char_length] = all_loaded_data[i]

        available_segments = {}
        for sid, sentence in enumerate(all_segment_level_char[0]):
            for wid, word in enumerate(sentence):
                segment = ''.join([CHARACTERS[i] for i in word])
                split_ids = np.asarray(np.nonzero(all_segment_level_term[0][sid][wid]))

                if segment in available_segments:
                    available_segments[segment].append([all_segment_level_stroke_out[0][sid][wid][:all_segment_level_stroke_length[0][sid][wid]], split_ids])
                else:
                    available_segments[segment] = [[all_segment_level_stroke_out[0][sid][wid][:all_segment_level_stroke_length[0][sid][wid]], split_ids]]

        index = 0
        all_W = []
        all_C = []

        # while index <= len(target_word):
        while index < len(target_word):
            available = False
            # Currently this just uses each character individually instead of the whole segment
            # for end_index in range(len(target_word), index, -1):
            #     segment = target_word[index:end_index]
            # print (segment)
            segment = target_word[index]
            if segment in available_segments:  # method beta
                # print(f'in dic - {segment}')
                available = True
                candidates = available_segments[segment]
                segment_level_stroke_out, split_ids = candidates[np.random.randint(len(candidates))]
                out = net.inf_state_fc1(torch.FloatTensor(segment_level_stroke_out).to(device).unsqueeze(0))
                out = net.inf_state_relu(out)
                seg_W_c, (h_n, _) = net.inf_state_lstm(out)

                character = segment[0]  # take the first character of the segment?

                # get character matrix using same method as method beta
                char_vector = torch.eye(len(CHARACTERS))[CHARACTERS.index(character)].to(device).unsqueeze(0)
                out = net.char_vec_fc_1(char_vector)
                out = net.char_vec_relu_1(out)
                out, _ = net.char_lstm_1(out.unsqueeze(0))
                out = out.squeeze(0)
                out = net.char_vec_fc2_1(out)
                char_matrix = out.view([-1, 256, 256])
                inv_char_matrix = char_matrix.inverse()

                id = split_ids[0][0]
                W_c_vector = seg_W_c[0, id].squeeze()

                # invert to get writer-independed DSD
                W_vector = torch.bmm(inv_char_matrix, W_c_vector.repeat(inv_char_matrix.size(0), 1).unsqueeze(2))
                all_W.append(W_vector)
                all_C.append(char_matrix)

                index += 1

            if index == len(target_word):
                break

            if not available:  # method alpha
                character = target_word[index]
                # print(f'no dic - {character}')
                char_vector = torch.eye(len(CHARACTERS))[CHARACTERS.index(character)].to(device).unsqueeze(0)
                out = net.char_vec_fc_1(char_vector)
                out = net.char_vec_relu_1(out)
                out, _ = net.char_lstm_1(out.unsqueeze(0))
                out = out.squeeze(0)
                out = net.char_vec_fc2_1(out)
                char_matrix = out.view([-1, 256, 256])

                W_vector = writer_mean_Ws[i].repeat(char_matrix.size(0), 1).unsqueeze(2)

                # all_W.append([W_vector])
                all_W.append(W_vector)
                all_C.append(char_matrix)

                index += 1

        all_writer_Ws[i, :, :] = torch.stack(all_W).squeeze()
        all_writer_Cs[i, :, :, :] = torch.stack(all_C).squeeze()

    return all_writer_Ws, all_writer_Cs


def get_commands(net, target_word, all_W_c):
    """converts character-dependent style-dependent DSDs to a list of commands for drawing"""
    all_commands = []
    current_id = 0
    while True:
        word_Wc_rec_TYPE_D = []
        TYPE_D_REF = []
        cid = 0
        for segment_batch_id in range(len(all_W_c)):
            if len(TYPE_D_REF) == 0:
                for each_segment_Wc in all_W_c[segment_batch_id]:
                    if cid >= current_id:
                        word_Wc_rec_TYPE_D.append(each_segment_Wc)
                    cid += 1
                if len(word_Wc_rec_TYPE_D) > 0:
                    TYPE_D_REF.append(all_W_c[segment_batch_id][-1])
            else:
                for each_segment_Wc in all_W_c[segment_batch_id]:
                    magic_inp = torch.cat([torch.stack(TYPE_D_REF, 0), each_segment_Wc.unsqueeze(0)], 0)
                    magic_inp = magic_inp.unsqueeze(0)
                    TYPE_D_out, (c, h) = net.magic_lstm(magic_inp)
                    TYPE_D_out = TYPE_D_out.squeeze(0)
                    word_Wc_rec_TYPE_D.append(TYPE_D_out[-1])
                TYPE_D_REF.append(all_W_c[segment_batch_id][-1])
        WC_ = torch.stack(word_Wc_rec_TYPE_D)
        tmp_commands, res = net.sample_from_w_fix(WC_, target_word)
        current_id = current_id + res
        if len(all_commands) == 0:
            all_commands.append(tmp_commands)
        else:
            all_commands.append(tmp_commands[1:])
        if res < 0 or current_id >= len(target_word):
            break

    commands = []
    px, py = 0, 100
    for coms in all_commands:
        for i, [dx, dy, t] in enumerate(coms):
            x = px + dx * 5
            y = py + dy * 5
            commands.append([x, y, t])
            px, py = x, y
    commands = np.asarray(commands)
    commands[:, 0] -= np.min(commands[:, 0])

    return commands


def sample(writer_weights, target_sentence, net, all_loaded_data, device="cpu"):
    """Generates an image of handwritten text based on target_sentence"""
    words = target_sentence.split(' ')

    im = Image.fromarray(np.zeros([160, 750]))
    dr = ImageDraw.Draw(im)
    width = 50

    writer_mean_Ws = []
    for loaded_data in all_loaded_data:
        mean_global_W = get_mean_global_W(net, loaded_data, device)
        writer_mean_Ws.append(mean_global_W)

    for word in words:
        all_writer_Ws, all_writer_Cs = get_DSD(net, word, writer_mean_Ws, all_loaded_data, device)
        all_W_c = get_writer_blend_W_c(writer_weights, all_writer_Ws, all_writer_Cs)
        all_commands = get_commands(net, word, all_W_c)

        for [x, y, t] in all_commands:
            if t == 0:
                dr.line((px+width, py, x+width, y), 255, 1)
            px, py = x, y
        width += np.max(all_commands[:, 0]) + 25

    return im


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for generating samples with the handwriting synthesis model.')

    parser.add_argument('--writer_id', type=int, default=80)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--generating_default', type=int, default=0)
    parser.add_argument('--direct_use', type=int, default=0)
    main(parser.parse_args())
