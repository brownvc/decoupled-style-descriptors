
import os
import re
from random import random
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
import ffmpeg # for problems with ffmpeg uninstall ffmpeg and then install ffmpeg-python

L = 256

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


def get_DSD(net, target_word, writer_mean_Ws, all_loaded_data, device):
    """
    returns a style vector and character matrix for each character/segment in target_word

    n is the number of writers
    M is the number of characters in the target word
    L is the latent vector size (in this case 256)

    input:
    - target_word, a string of length M to be converted to a DSD
    - writer_mean_Ws, a list of n style vectors of size L

    output:
    - all_writer_Ws, a tensor of size n x M x L representing the style vectors for each writer and character 
    - all_writer_Cs, a tensor of size n x M x L x L representing the corresponding character matrix
    """

    n = len(all_loaded_data)
    M = len(target_word)
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


def get_writer_blend_W_c(writer_weights, all_Ws, all_Cs):
    """
    generates character-dependent style-dependent DSDs for each character/segement in target_word,
    averaging together the styles of the handwritings using provided weights

    n is the number of writers
    M is the number of characters in the target word
    L is the latent vector size (in this case 256)

    input:
    - writer_weights, a list of length n weights for each writer that sum to one
    - all_writer_Ws, an n x M x L tensor representing each weiter's style vector for every character
    - all_writer_Cs, an n x M x L x L tensor representing the style's correspodning character matrix

    output:
    - an M x 1 x L tensor of M scharacter-dependent style-dependent DSDs
    """
    n, M, _ = all_Ws.shape
    weights_tensor = torch.tensor(writer_weights).repeat_interleave(M * L).reshape(n, M, L)  # repeat accross remaining dimensions
    W_vectors = (weights_tensor * all_Ws).sum(axis=0).unsqueeze(-1)  # take weighted sum accross writers axis
    char_matrices = all_Cs[0, :, :, :]  # character matrices are independent of writer

    W_cs = torch.bmm(char_matrices, W_vectors)

    return W_cs.reshape(M, 1, L)


def get_character_blend_W_c(character_weights, all_Ws, all_Cs):
    """
    generates a single character-dependent style-dependent DSD,
    averaging together the characters using provided weights

    M is the number of characters to blend
    L is the latent vector size (in this case 256)

    input:
    - character_weights, a list of length M weights for each character that sum to one
    - all_Ws, a 1 x M x L tensor representing the wwiter's style vector for each character
    - all_Cs, 1 x M x L x L tensor representing the style's correspodning character matrix

    output:
    - a 1 x 1 x L tensor representing the character-dependent style-dependent DSDs
    """
    M = len(character_weights)
    W_vector = all_Ws[0, 0, :].unsqueeze(-1)

    weights_tensor = torch.tensor(character_weights).repeat_interleave(L * L).reshape(1, M, L, L)  # repeat accross remaining dimensions
    char_matrix = (weights_tensor * all_Cs).sum(axis=1).squeeze() # take weighted sum accross characters axis

    W_c = char_matrix @ W_vector

    return W_c.reshape(1, 1, L)


def get_commands(net, target_word, all_W_c): # seems like target_word is only used for length
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
        tmp_commands, res = net.sample_from_w_fix(WC_)
        current_id += res
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

def mdn_video(target_word, num_samples, scale_sd, clamp_mdn, net, all_loaded_data, device):
    '''
    Method creating gif of mdn samples
    num_samples: number of samples to be inputted
    max_scale: the maximum value used to scale SD while sampling (increment is based on num samples)
    '''
    words = target_word.split(' ')
    us_target_word = re.sub(r"\s+", '_', target_word)
    os.makedirs(f"./results/{us_target_word}_mdn_samples", exist_ok=True)
    for i in range(num_samples):
        im = Image.fromarray(np.zeros([160, 750]))
        dr = ImageDraw.Draw(im)
        width = 50

        net.scale_sd = scale_sd
        net.clamp_mdn = clamp_mdn

        mean_global_W = get_mean_global_W(net, all_loaded_data[0], device)

        for word in words:
            writer_Ws, writer_Cs = get_DSD(net, word, [mean_global_W], [all_loaded_data[0]], device)
            all_W_c = get_writer_blend_W_c([1], writer_Ws, writer_Cs)
            all_commands = get_commands(net, word, all_W_c)

            for [x, y, t] in all_commands:
                if t == 0:
                    dr.line((px+width, py, x+width, y), 255, 1)
                px, py = x, y
            width += np.max(all_commands[:, 0]) + 25

        im.convert("RGB").save(f'results/{us_target_word}_mdn_samples/sample_{i}.png')
    # Convert fromes to video using ffmpeg
    photos = ffmpeg.input(f'results/{us_target_word}_mdn_samples/sample_*.png', pattern_type='glob', framerate=10)
    videos = photos.output(f'results/{us_target_word}_video.mov', vcodec="libx264", pix_fmt="yuv420p")
    videos.run(overwrite_output=True)

def sample_blended_writers(writer_weights, target_sentence, net, all_loaded_data, device="cpu"):
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


def sample_character_grid(letters, grid_size, net, all_loaded_data, device="cpu"):
    """Generates an image of handwritten text based on target_sentence"""
    width = 60
    im = Image.fromarray(np.zeros([(grid_size + 1) * width, (grid_size + 1) * width]))
    dr = ImageDraw.Draw(im)

    M = len(letters)
    mean_global_W = get_mean_global_W(net, all_loaded_data[0], device)

    # all_Ws = torch.zeros(1, M, L)
    all_Cs = torch.zeros(1, M, L, L)
    for i in range(M):  # get corners of grid
        W_vector, char_matrix = get_DSD(net, letters[i], [mean_global_W], [all_loaded_data[0]], device)
        # all_Ws[:, i, :] = W_vector
        all_Cs[:, i, :, :] = char_matrix

    all_Ws = mean_global_W.reshape(1, 1, L)

    for i in range(grid_size):
        for j in range(grid_size):
            wx = i / (grid_size - 1)
            wy = j / (grid_size - 1)

            character_weights = [(1 - wx) * (1 - wy), # top left is 1 at (0, 0)
                                 wx       * (1 - wy), # top right is 1  at (1, 0)
                                 (1 - wx) * wy,       # bottom left is 1 at (0, 1)
                                 wx       * wy]       # bottom right is 1 at (1, 1)
            all_W_c = get_character_blend_W_c(character_weights, all_Ws, all_Cs)
            all_commands = get_commands(net, letters[0], all_W_c)

            offset_x = i * width
            offset_y = j * width

            for [x, y, t] in all_commands:
                if t == 0:
                    dr.line((
                        px + offset_x + width/2,
                        py + offset_y - width/2,  # letters are shifted down for some reason
                        x + offset_x + width/2,
                        y + offset_y - width/2), 255, 1)
                px, py = x, y

    return im

def writer_interpolation_video(target_sentence, transition_time, net, all_loaded_data, device="cpu"):
    """
    Generates a video of interpolating between each provided writer
    """

    n = len(all_loaded_data)

    os.makedirs(f"./results/{target_sentence}_blend_frames", exist_ok=True)

    words = target_sentence.split(' ')

    writer_mean_Ws = []
    for loaded_data in all_loaded_data:
        mean_global_W = get_mean_global_W(net, loaded_data, device)
        writer_mean_Ws.append(mean_global_W)

    word_Ws = []
    word_Cs = []

    for word in words:
        all_writer_Ws, all_writer_Cs = get_DSD(net, word, writer_mean_Ws, all_loaded_data, device)
        word_Ws.append(all_writer_Ws)
        word_Cs.append(all_writer_Cs)

    for i in range(n - 1):
        for j in range(transition_time):
            im = Image.fromarray(np.zeros([160, 750]))
            dr = ImageDraw.Draw(im)
            width = 50

            completion = j/(transition_time)

            individual_weights = [1 - completion, completion]
            writer_weights = [0] * i + individual_weights + [0] * (n - 2 - i)

            for k, word in enumerate(words):
                all_writer_Ws, all_writer_Cs = word_Ws[k], word_Cs[k]
                all_W_c = get_writer_blend_W_c(writer_weights, all_writer_Ws, all_writer_Cs)
                all_commands = get_commands(net, word, all_W_c)

                for [x, y, t] in all_commands:
                    if t == 0:
                        dr.line((px+width, py, x+width, y), 255, 1)
                    px, py = x, y
                width += np.max(all_commands[:, 0]) + 25

            im.convert("RGB").save(f"./results/{target_sentence}_blend_frames/frame_{str(i * transition_time + j).zfill(3)}.png")

    # Convert fromes to video using ffmpeg
    photos = ffmpeg.input(f"./results/{target_sentence}_blend_frames/frame_*.png", pattern_type='glob', framerate=10)
    videos = photos.output(f"results/{target_sentence}_blend_video.mov", vcodec="libx264", pix_fmt="yuv420p")
    videos.run(overwrite_output=True)

def mdn_single_sample(target_word, scale_sd, clamp_mdn, net, all_loaded_data, device):
    '''
    Method creating gif of mdn samples
    num_samples: number of samples to be inputted
    max_scale: the maximum value used to scale SD while sampling (increment is based on num samples)
    '''
    words = target_word.split(' ')
    im = Image.fromarray(np.zeros([160, 750]))
    dr = ImageDraw.Draw(im)
    width = 50

    net.scale_sd = scale_sd
    net.clamp_mdn = clamp_mdn

    mean_global_W = get_mean_global_W(net, all_loaded_data[0], device)

    for word in words:
        writer_Ws, writer_Cs = get_DSD(net, word, [mean_global_W], [all_loaded_data[0]], device)
        all_W_c = get_writer_blend_W_c([1], writer_Ws, writer_Cs)
        all_commands = get_commands(net, word, all_W_c)

        for [x, y, t] in all_commands:
            if t == 0:
                dr.line((px+width, py, x+width, y), 255, 1)
            px, py = x, y
        width += np.max(all_commands[:, 0]) + 25

    return im

def sample_blended_chars(character_weights, letters, net, all_loaded_data, device="cpu"):
    """Generates an image of handwritten text based on target_sentence"""

    width = 60
    im = Image.fromarray(np.zeros([100, 100]))
    dr = ImageDraw.Draw(im)

    M = len(letters)
    mean_global_W = get_mean_global_W(net, all_loaded_data[0], device)

    all_Cs = torch.zeros(1, M, L, L)
    for i in range(M):  # get corners of grid
        W_vector, char_matrix = get_DSD(net, letters[i], [mean_global_W], [all_loaded_data[0]], device)
        all_Cs[:, i, :, :] = char_matrix

    all_Ws = mean_global_W.reshape(1, 1, L)

    all_W_c = get_character_blend_W_c(character_weights, all_Ws, all_Cs)
    all_commands = get_commands(net, letters[0], all_W_c)

    for [x, y, t] in all_commands:
        if t == 0:
            dr.line((
                px + width/2,
                py - width/2,  # letters are shifted down for some reason
                x + width/2,
                y - width/2), 255, 1)
        px, py = x, y

        
    return im


def char_interpolation_video(letters, transition_time, net, all_loaded_data, device="cpu"):
    """Generates an image of handwritten text based on target_sentence"""

    os.makedirs(f"./results/{''.join(letters)}_frames", exist_ok=True) # make a folder for the frames

    width = 50

    M = len(letters)
    mean_global_W = get_mean_global_W(net, all_loaded_data[0], device)

    all_Cs = torch.zeros(1, M, L, L)
    for i in range(M):  # get corners of grid
        W_vector, char_matrix = get_DSD(net, letters[i], [mean_global_W], [all_loaded_data[0]], device)
        all_Cs[:, i, :, :] = char_matrix

    all_Ws = mean_global_W.reshape(1, 1, L)

    for i in range(M - 1):
        for j in range(transition_time):
            completion = j / (transition_time - 1)
            individual_weights = [1 - completion, completion]
            character_weights = [0] * i + individual_weights + [0] * (M - 2 - i)
            all_W_c = get_character_blend_W_c(character_weights, all_Ws, all_Cs)
            all_commands = get_commands(net, "change this later!", all_W_c)

            im = Image.fromarray(np.zeros([100, 100]))
            dr = ImageDraw.Draw(im)

            for [x, y, t] in all_commands:
                if t == 0:
                    dr.line((
                        px + width/2,
                        py - width/2,  # letters are shifted down for some reason
                        x + width/2,
                        y - width/2), 255, 1)
                px, py = x, y

                
            im.convert("RGB").save(f"results/{''.join(letters)}_frames/frames_{str(i * transition_time + j).zfill(3)}.png")

    # Convert fromes to video using ffmpeg
    photos = ffmpeg.input(f"results/{''.join(letters)}_frames/frames_*.png", pattern_type='glob', framerate=24)
    videos = photos.output(f"results/{''.join(letters)}_video.mov", vcodec="libx264", pix_fmt="yuv420p")
    videos.run(overwrite_output=True)


