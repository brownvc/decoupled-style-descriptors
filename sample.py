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
	dl = DataLoader(num_writer=1, num_samples=10, divider=5.0, datadir='./data/writers')
	net	= SynthesisNetwork(weight_dim=256, num_layers=3).to(device)

	if not torch.cuda.is_available():
		net.load_state_dict(torch.load('./model/250000.pt', map_location=torch.device('cpu')))

	[_, _, _, _, _, _, all_word_level_stroke_in, all_word_level_stroke_out, all_word_level_stroke_length, all_word_level_term, all_word_level_char, all_word_level_char_length, all_segment_level_stroke_in, all_segment_level_stroke_out, all_segment_level_stroke_length, all_segment_level_term,all_segment_level_char, all_segment_level_char_length] = dl.next_batch(TYPE='TRAIN', uid=params.writer_id, tids=[i for i in range(params.num_samples)])

	batch_word_level_stroke_in 			= [torch.FloatTensor(a).to(device) for a in all_word_level_stroke_in]
	batch_word_level_stroke_out 		= [torch.FloatTensor(a).to(device) for a in all_word_level_stroke_out]
	batch_word_level_stroke_length 		= [torch.LongTensor(a).to(device).unsqueeze(-1) for a in all_word_level_stroke_length]
	batch_word_level_term 				= [torch.FloatTensor(a).to(device) for a in all_word_level_term]
	batch_word_level_char 				= [torch.LongTensor(a).to(device) for a in all_word_level_char]
	batch_word_level_char_length 		= [torch.LongTensor(a).to(device).unsqueeze(-1) for a in all_word_level_char_length]
	batch_segment_level_stroke_in 		= [[torch.FloatTensor(a).to(device) for a in b] for b in all_segment_level_stroke_in]
	batch_segment_level_stroke_out 		= [[torch.FloatTensor(a).to(device) for a in b] for b in all_segment_level_stroke_out]
	batch_segment_level_stroke_length 	= [[torch.LongTensor(a).to(device).unsqueeze(-1) for a in b] for b in all_segment_level_stroke_length]
	batch_segment_level_term 			= [[torch.FloatTensor(a).to(device) for a in b] for b in all_segment_level_term]
	batch_segment_level_char 			= [[torch.LongTensor(a).to(device) for a in b] for b in all_segment_level_char]
	batch_segment_level_char_length 	= [[torch.LongTensor(a).to(device).unsqueeze(-1) for a in b] for b in all_segment_level_char_length]

	if params.generating_default == 1:
		with torch.no_grad():
			commands_list = net.sample([ batch_word_level_stroke_in, batch_word_level_stroke_out, batch_word_level_stroke_length, batch_word_level_term, batch_word_level_char, batch_word_level_char_length, batch_segment_level_stroke_in, batch_segment_level_stroke_out, batch_segment_level_stroke_length, batch_segment_level_term, batch_segment_level_char, batch_segment_level_char_length])
		[t_commands, o_commands, a_commands, b_commands, c_commands, d_commands] = commands_list
		dst = Image.new('RGB', (750, 640))
		dst.paste(draw_commands(t_commands), (0, 0))
		dst.paste(draw_commands(o_commands), (0, 160))
		dst.paste(draw_commands(a_commands), (0, 320))
		dst.paste(draw_commands(d_commands), (0, 480))
		dst.save(f'results/default.png')

	with torch.no_grad():
		word_inf_state_out 		= net.inf_state_fc1(batch_word_level_stroke_out[0])
		word_inf_state_out		= net.inf_state_relu(word_inf_state_out)
		word_inf_state_out, _ 	= net.inf_state_lstm(word_inf_state_out)

		user_word_level_char	= batch_word_level_char[0]
		user_word_level_term	= batch_word_level_term[0]

		original_Wc				= []
		word_batch_id 			= 0

		curr_seq_len 			= batch_word_level_stroke_length[0][word_batch_id][0]
		curr_char_len			= batch_word_level_char_length[0][word_batch_id][0]

		char_vector				= torch.eye(len(CHARACTERS))[user_word_level_char[word_batch_id][:curr_char_len]].to(device)
		current_term			= user_word_level_term[word_batch_id][:curr_seq_len].unsqueeze(-1)
		split_ids				= torch.nonzero(current_term)[:,0]

		char_vector_1			= net.char_vec_fc_1(char_vector)
		char_vector_1			= net.char_vec_relu_1(char_vector_1)

		char_out_1				= char_vector_1.unsqueeze(0)
		char_out_1, (c,h) 		= net.char_lstm_1(char_out_1)
		char_out_1 				= char_out_1.squeeze(0)
		char_out_1				= net.char_vec_fc2_1(char_out_1)
		char_matrix_1			= char_out_1.view([-1,1,256,256])
		char_matrix_1			= char_matrix_1.squeeze(1)
		char_matrix_inv_1		= torch.inverse(char_matrix_1)

		W_c_t					= word_inf_state_out[word_batch_id][:curr_seq_len]
		W_c						= torch.stack([W_c_t[i] for i in split_ids])
		original_Wc.append(W_c)

		W						= torch.bmm(char_matrix_inv_1, W_c.unsqueeze(2)).squeeze(-1)
		mean_global_W			= torch.mean(W, 0)


	def sample_word(target_word):
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
		all_W_c = []

		while index <= len(target_word):
			available = False
			for end_index in range(len(target_word), index, -1):
				segment = target_word[index:end_index]
				# print (segment)

				if segment in available_segments:
					# print (f'in dic - {segment}')
					available = True
					candidates = available_segments[segment]
					segment_level_stroke_out, split_ids = candidates[np.random.randint(len(candidates))]
					out = net.inf_state_fc1(torch.FloatTensor(segment_level_stroke_out).to(device).unsqueeze(0))
					out = net.inf_state_relu(out)
					seg_W_c, _ = net.inf_state_lstm(out)
					tmp = []
					for id in split_ids[0]:
						tmp.append(seg_W_c[0, id].squeeze())
					all_W_c.append(tmp)
					index = end_index

			if index == len(target_word):
				break

			if not available:
				character = target_word[index]
				# print (f'no dic - {character}')
				char_vector = torch.eye(len(CHARACTERS))[CHARACTERS.index(character)].to(device).unsqueeze(0)
				out = net.char_vec_fc_1(char_vector)
				out = net.char_vec_relu_1(out)
				out, _ = net.char_lstm_1(out.unsqueeze(0))
				out = out.squeeze(0)
				out = net.char_vec_fc2_1(out)
				char_matrix = out.view([-1, 256, 256])
				TYPE_A_WC	= torch.bmm(char_matrix, mean_global_W.repeat(char_matrix.size(0), 1).unsqueeze(2)).squeeze()
				all_W_c.append([TYPE_A_WC])
				index += 1

		
		all_commands = []
		current_id = 0
		while True:
			word_Wc_rec_TYPE_D		= []
			TYPE_D_REF				= []
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
						magic_inp 	= torch.cat([torch.stack(TYPE_D_REF, 0), each_segment_Wc.unsqueeze(0)], 0)
						magic_inp	= magic_inp.unsqueeze(0)
						TYPE_D_out, (c,h) = net.magic_lstm(magic_inp)
						TYPE_D_out = TYPE_D_out.squeeze(0)
						word_Wc_rec_TYPE_D.append(TYPE_D_out[-1])
					TYPE_D_REF.append(all_W_c[segment_batch_id][-1])
			WC_	= torch.stack(word_Wc_rec_TYPE_D)
			tmp_commands, res = net.sample_from_w_fix(WC_, target_word)
			current_id = current_id + res
			if len(all_commands) == 0:
				all_commands.append(tmp_commands)
			else:
				all_commands.append(tmp_commands[1:])
			if res < 0 or current_id >= len(target_word):
				break

		# tmp_commands = net.sample_from_w_fix(torch.stack(tmp_WC), _, target_word)

		commands = []
		px, py = 0, 100
		for coms in all_commands:
			for i, [dx, dy, t] in enumerate(coms):
				x = px + dx * 5
				y = py + dy * 5
				commands.append([x,y,t])
				px, py = x, y
		commands = np.asarray(commands)
		commands[:, 0] -= np.min(commands[:, 0])

		return commands

	def sample(target_sentence):
		words = target_sentence.split(' ')

		im = Image.fromarray(np.zeros([160, 750]))
		dr = ImageDraw.Draw(im)
		width = 50

		for word in words:
			all_commands = sample_word(word)

			for [x,y,t] in all_commands:
				if t == 0:
					dr.line((px+width, py, x+width, y), 255, 1)
				px, py = x, y
			width += np.max(all_commands[:, 0]) + 25

		# im.convert("RGB").save(f'results/{target_word}.png')
		im.convert("RGB").save(f'results/hello.png')

	def sample_word2(target_word):
		available_segments = {}
		for sid, sentence in enumerate(all_segment_level_char[0]):
			for wid, word in enumerate(sentence):
				segment = ''.join([CHARACTERS[i] for i in word])
				split_ids = np.asarray(np.nonzero(all_segment_level_term[0][sid][wid]))

				if segment in available_segments:
					available_segments[segment].append([all_segment_level_stroke_in[0][sid][wid][:all_segment_level_stroke_length[0][sid][wid]], all_segment_level_stroke_out[0][sid][wid][:all_segment_level_stroke_length[0][sid][wid]], split_ids])
				else:
					available_segments[segment] = [[all_segment_level_stroke_in[0][sid][wid][:all_segment_level_stroke_length[0][sid][wid]], all_segment_level_stroke_out[0][sid][wid][:all_segment_level_stroke_length[0][sid][wid]], split_ids]]

		index = 0
		all_W_c = []

		all_commands = []
		know_chars = []
		while index <= len(target_word):
			available = False
			for end_index in range(len(target_word), index, -1):
				segment = target_word[index:end_index]
				if segment in available_segments:
					# print (f'in dic - {segment}')
					available = True
					candidates = available_segments[segment]
					segment_level_stroke_in, segment_level_stroke_out, split_ids = candidates[np.random.randint(len(candidates))]
					out = net.inf_state_fc1(torch.FloatTensor(segment_level_stroke_in).to(device).unsqueeze(0))
					out = net.inf_state_relu(out)
					seg_W_c, _ = net.inf_state_lstm(out)
					tmp = []
					for id in split_ids[0]:
						# print (id)
						tmp.append(seg_W_c[0, id].squeeze())
					all_W_c.append(tmp)
					index = end_index
					for i in range(index, end_index):
						know_chars.append(i)

					commands = []
					px, py = 0, 100
					for i, [dx, dy, t] in enumerate(segment_level_stroke_out):
						x = px + dx * 5
						y = py + dy * 5
						commands.append([x,y,t])
						px, py = x, y
					commands = np.asarray(commands)
					commands[:, 0] -= np.min(commands[:, 0])
					all_commands.append(commands)

			if index == len(target_word):
				break

			if not available:
				character = target_word[index]
				# print (f'no dic - {character}')
				char_vector = torch.eye(len(CHARACTERS))[CHARACTERS.index(character)].to(device).unsqueeze(0)
				out = net.char_vec_fc_1(char_vector)
				out = net.char_vec_relu_1(out)
				out, _ = net.char_lstm_1(out.unsqueeze(0))
				out = out.squeeze(0)
				out = net.char_vec_fc2_1(out)
				char_matrix = out.view([-1, 256, 256])
				TYPE_A_WC	= torch.bmm(char_matrix, mean_global_W.repeat(char_matrix.size(0), 1).unsqueeze(2)).squeeze().unsqueeze(0)
				index += 1

				temp_commands = net.sample_from_w(TYPE_A_WC, character)

				commands = []
				px, py = 0, 100
				for i, [dx, dy, t] in enumerate(temp_commands):
					x = px + dx * 5
					y = py + dy * 5
					commands.append([x,y,t])
					px, py = x, y
				commands = np.asarray(commands)
				commands[:, 0] -= np.min(commands[:, 0])
				all_commands.append(commands)
		return all_commands

	def sample2(target_sentence):
		words = target_sentence.split(' ')

		im = Image.fromarray(np.zeros([160, 750]))
		dr = ImageDraw.Draw(im)
		width = 50

		for word in words:
			all_commands = sample_word2(word)

			for c in all_commands:
				for [x,y,t] in c:
					if t == 0:
						dr.line((px+width, py, x+width, y), 255, 1)
					px, py = x, y
				width += np.max(c[:, 0]) + 5
			
			width += 25

		# im.convert("RGB").save(f'results/{target_word}.png')
		im.convert("RGB").save(f'results/hello2.png')

	while True:
		target_word = input("Type a sentence to generate : ")
		if len(target_word) > 0:
			if params.direct_use == 0:
				sample(target_word)
			else:
				sample2(target_word)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Arguments for generating samples with the handwriting synthesis model.')

	parser.add_argument('--writer_id', type=int, default=80)
	parser.add_argument('--num_samples', type=int, default=10)
	parser.add_argument('--generating_default', type=int, default=0)
	parser.add_argument('--direct_use', type=int, default=0)
	main(parser.parse_args())

