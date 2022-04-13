from config.GlobalVariables import *
import math
import torch
from scipy import stats
import numpy as np
from PIL import Image, ImageDraw
import os
import pickle

def preprocess_dataset(data_dir, resample=20, pred_start=1):
    def reformat_raw_data(raw_data, pred_start):
        if pred_start == 1:
            tmp = np.concatenate([[[0, 500, 0]], raw_data], 0)
            tmp = tmp[1:] - tmp[:-1]
            tmp[1:, 2] = raw_data[:-1, 2]
            tmp = np.concatenate([[[0, 0, 0]], tmp], 0)
        else:
            tmp = np.concatenate([raw_data[0:1], raw_data])
            tmp = tmp[1:] - tmp[:-1]
            tmp[0,2] = 0
            tmp[1:,2] = raw_data[:-1, 2]
        return tmp[:-1], tmp[1:]

    prohibits = [f'./BRUSH/5/118_resample{resample}', f'./BRUSH/7/14_resample{resample}',
                 f'./BRUSH/7/101_resample{resample}', f'./BRUSH/7/58_resample{resample}',
                 f'./BRUSH/14/20_resample{resample}', f'./BRUSH/22/45_resample{resample}',
                 f'./BRUSH/30/45_resample{resample}', f'./BRUSH/40/85_resample{resample}',
                 f'./BRUSH/50/45_resample{resample}', f'./BRUSH/59/29_resample{resample}',
                 f'./BRUSH/96/120_resample{resample}', f'./BRUSH/99/134_resample{resample}',
                 f'./BRUSH/140/35_resample{resample}', f'./BRUSH/144/55_resample{resample}',
                 f'./BRUSH/144/91_resample{resample}', f'./BRUSH/144/28_resample{resample}',
                 f'./BRUSH/144/69_resample{resample}']

    preprocess_dir = 'preprocess' if pred_start == 1 else 'preprocess2'

    for writer_id in range(170):
        print(f'Preprocessing the BRUSH dataset - Finished Writer ID: {writer_id + 1} / 170')
        for sentence_id in [i for i in os.listdir(f'{data_dir}/{writer_id}') if i[-3:] == f'e{resample}']:
            with open(f'{data_dir}/{writer_id}/{sentence_id}', 'rb') as f:
                [sentence_text, raw_points, character_labels] = pickle.load(f)
            if f'{data_dir}/{writer_id}/{sentence_id}' not in prohibits:
                sentence_raw_points = raw_points
                sentence_raw_points[:, 0] -= sentence_raw_points[0, 0]
                sentence_stroke_in, sentence_stroke_out = reformat_raw_data(sentence_raw_points, pred_start=pred_start)

                split_char_ids = [i for i, c in enumerate(sentence_text) if c == ' ']

                sentence_char = [CHARACTERS.find(c) for c in sentence_text]

                sentence_term = []
                cid = 0
                for i in range(len(character_labels) - 1):
                    if character_labels[i + 1, cid] != 1:
                        if np.argmax(character_labels[i + 1]) >= cid:
                            cid += 1
                            sentence_term.append(1)
                        else:
                            sentence_term.append(0)
                    else:
                        sentence_term.append(0)
                sentence_term.append(1)
                sentence_term = np.asarray(sentence_term)

                assert (len(sentence_term) == len(character_labels))

                word_level_raw_stroke = []
                word_level_stroke_in = []
                word_level_stroke_out = []
                word_level_char = []
                word_level_term = []

                segment_level_raw_stroke = []
                segment_level_stroke_in = []
                segment_level_stroke_out = []
                segment_level_char = []
                segment_level_term = []

                character_level_raw_stroke = []
                character_level_stroke_in = []
                character_level_stroke_out = []
                character_level_char = []
                character_level_term = []

                word_start_id = 0

                for i, c in enumerate(sentence_text):
                    if c != ' ':
                        character_raw_points = raw_points[character_labels[:, i] > 0]
                        character_raw_points[:, 0] -= character_raw_points[0, 0]
                        character_level_raw_stroke.append(character_raw_points)
                        character_stroke_in, character_stroke_out = reformat_raw_data(character_raw_points, pred_start=pred_start)
                        character_level_stroke_in.append(character_stroke_in)
                        character_level_stroke_out.append(character_stroke_out)
                        term = np.zeros([len(character_raw_points)])
                        term[-1] = 1
                        character_level_term.append(term)
                        character_level_char.append([CHARACTERS.find(c)])

                    if i in split_char_ids:
                        word = sentence_text[word_start_id:i]
                        word_labels = np.zeros(len(character_labels))
                        for j in range(word_start_id, i):
                            word_labels += character_labels[:, j]
                        word_raw_points = raw_points[word_labels > 0]
                        word_term = sentence_term[word_labels > 0]
                        word_term[0] = 0
                        assert (np.sum(word_term) == len(word))
                        word_raw_points[:, 0] -= word_raw_points[0, 0]
                        word_level_raw_stroke.append(word_raw_points)
                        word_stroke_in, word_stroke_out = reformat_raw_data(word_raw_points, pred_start=pred_start)
                        word_level_stroke_in.append(word_stroke_in)
                        word_level_stroke_out.append(word_stroke_out)
                        word_level_term.append(word_term)
                        word_level_char.append([CHARACTERS.find(c) for c in word])
                        word_start_id = i + 1

                        assert (len(character_level_raw_stroke) == len(word))

                        segment_level_raw_stroke.append(character_level_raw_stroke)
                        segment_level_stroke_in.append(character_level_stroke_in)
                        segment_level_stroke_out.append(character_level_stroke_out)
                        segment_level_char.append(character_level_char)
                        segment_level_term.append(character_level_term)
                        character_level_raw_stroke = []
                        character_level_stroke_in = []
                        character_level_stroke_out = []
                        character_level_char = []
                        character_level_term = []

                word = sentence_text[word_start_id:]
                word_labels = np.zeros(len(character_labels))
                for j in range(word_start_id, len(sentence_text)):
                    word_labels += character_labels[:, j]
                word_raw_points = raw_points[word_labels > 0]
                word_raw_points[:, 0] -= word_raw_points[0, 0]
                word_term = sentence_term[word_labels > 0]
                word_term[0] = 0
                assert (np.sum(word_term) == len(word))
                word_level_raw_stroke.append(word_raw_points)
                word_stroke_in, word_stroke_out = reformat_raw_data(word_raw_points, pred_start=pred_start)
                word_level_stroke_in.append(word_stroke_in)
                word_level_stroke_out.append(word_stroke_out)
                word_level_term.append(word_term)
                word_level_char.append([CHARACTERS.find(c) for c in word])
                assert (len(character_level_raw_stroke) == len(word))
                segment_level_raw_stroke.append(character_level_raw_stroke)
                segment_level_stroke_in.append(character_level_stroke_in)
                segment_level_stroke_out.append(character_level_stroke_out)
                segment_level_char.append(character_level_char)
                segment_level_term.append(character_level_term)

                if not os.path.exists(f'{data_dir}/{preprocess_dir}/{writer_id}'):
                    os.mkdir(f'{data_dir}/{preprocess_dir}/{writer_id}')

                with open(f'{data_dir}/{preprocess_dir}/{writer_id}/{sentence_id}', 'wb') as f:
                    pickle.dump([
                        sentence_stroke_in, sentence_stroke_out, sentence_term, sentence_char,
                        word_level_stroke_in, word_level_stroke_out, word_level_term, word_level_char,
                        segment_level_stroke_in, segment_level_stroke_out, segment_level_term, segment_level_char], f)


def gaussian_2d(x1, x2, mu1, mu2, s1, s2, rho):
    norm1 = x1 - mu1
    norm2 = x2 - mu2
    s1s2 = s1 * s2
    z = (norm1 / s1) ** 2 + (norm2 / s2) ** 2 - 2 * rho * norm1 * norm2 / s1s2
    numerator = torch.exp(-z / (2 * (1 - rho ** 2)))
    denominator = 2 * math.pi * s1s2 * torch.sqrt(1 - rho ** 2)
    gaussian = numerator / denominator
    return gaussian


def get_minimax(stroke_results):
    minimas = []
    maximas = []
    for stroke in stroke_results:
        for i, [x, y] in enumerate(stroke):
            if i == 0:
                prev_x, prev_y = x, y
            if i == len(stroke) - 1:
                if prev_y <= y:
                    maximas.append([x, y])
                if prev_y >= y:
                    minimas.append([x, y])
                break
            else:
                next_x, next_y = stroke[i + 1]
                if prev_y <= y and y >= next_y:
                    maximas.append([x, y])
                if prev_y >= y and y <= next_y:
                    minimas.append([x, y])

    minimas = np.asarray(minimas)
    maximas = np.asarray(maximas)
    return minimas, maximas


def get_slope(minimas, maximas):
    minima_slope, minima_intercept, _, _, _ = stats.linregress(minimas[:, 0], minimas[:, 1])
    maxima_slope, maxima_intercept, _, _, _ = stats.linregress(maximas[:, 0], maximas[:, 1])
    min_se = []
    max_se = []
    for [x, y] in minimas:
        min_ny = minima_slope * x + minima_intercept
        min_se.append(abs(min_ny - y))
    for [x, y] in maximas:
        max_ny = maxima_slope * x + maxima_intercept
        max_se.append(abs(max_ny - y))
    min_se, max_se = np.asarray(min_se), np.asarray(max_se)
    new_minimas = minimas[min_se < np.mean(min_se)]
    new_maximas = maximas[max_se < np.mean(max_se)]
    if len(new_minimas) > 5:
        minima_slope, minima_intercept, _, _, _ = stats.linregress(new_minimas[:, 0], new_minimas[:, 1])
    if len(new_maximas) > 5:
        maxima_slope, maxima_intercept, _, _, _ = stats.linregress(new_maximas[:, 0], new_maximas[:, 1])

    return minima_slope, minima_intercept, maxima_slope, maxima_intercept


def draw_commands(commands):
    im = Image.fromarray(np.zeros([160, 750]))
    dr = ImageDraw.Draw(im)

    px, py = 50, 100
    for i, [dx, dy, t] in enumerate(commands):
        x = px + dx * 5
        y = py + dy * 5
        if t == 0:
            dr.line((px, py, x, y), 255, 1)
        px, py = x, y

    return im


def draw_points(raw_points, character_labels):
    [w, h, _] = np.max(raw_points, 0)
    im = Image.new("RGB", [int(w) + 100, int(h) + 100])
    dr = ImageDraw.Draw(im)

    colors = np.random.randint(0, 255, (len(character_labels[0]), 3))

    for i, [x, y, t] in enumerate(raw_points):
        if i > 0:
            if pt == 0:
                dr.line((px, py, x, y), tuple(colors[np.argmax(character_labels[i])]), 3)
        px, py, pt = x, y, t

    return im
