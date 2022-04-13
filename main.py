# python -u main.py --divider 5.0 --weight_dim 256 --sample 5 --device 0 --num_layers 3 --num_writer 1 --lr 0.001 --VALIDATION 1 --datadir 2 --TYPE_B 0 --TYPE_C 0

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
from PIL import Image, ImageDraw, ImageFont
from DataLoader import DataLoader
import pickle
from config.GlobalVariables import *
import os
import argparse
from SynthesisNetwork import SynthesisNetwork

def main(params):
	cwds = os.getcwd()
	cwd = cwds.split('/')[-1]

	divider		= params.divider
	weight_dim	= params.weight_dim
	num_samples	= params.sample
	did			= params.device
	num_layers	= params.num_layers
	num_writer	= params.num_writer
	lr			= params.lr
	no_char		= params.no_char

	datadir = './data/writers'

	if params.VALIDATION == 1:
		VALIDATION = True
	else:
		VALIDATION = False

	if params.sentence_loss == 1:
		sentence_loss = True
		writer_sentence = SummaryWriter(logdir='./runs/sentence-' + cwd)
		if VALIDATION:
			valid_writer_sentence = SummaryWriter(logdir='./runs/valid-sentence-' + cwd)
	else:
		sentence_loss = False
	if params.word_loss == 1:
		word_loss = True
		writer_word = SummaryWriter(logdir='./runs/word-' + cwd)
		if VALIDATION:
			valid_writer_word = SummaryWriter(logdir='./runs/valid-word-' + cwd)
	else:
		word_loss = False
	if params.segment_loss == 1:
		segment_loss = True
		writer_segment = SummaryWriter(logdir='./runs/segment-' + cwd)
		if VALIDATION:
			valid_writer_segment = SummaryWriter(logdir='./runs/valid-segment-' + cwd)
	else:
		segment_loss = False
	if params.TYPE_A == 1:
		TYPE_A = True
	else:
		TYPE_A = False
	if params.TYPE_B == 1:
		TYPE_B = True
	else:
		TYPE_B = False
	if params.TYPE_C == 1:
		TYPE_C = True
	else:
		TYPE_C = False
	if params.TYPE_D == 1:
		TYPE_D = True
	else:
		TYPE_D = False
	if params.ORIGINAL == 1:
		ORIGINAL = True
	else:
		ORIGINAL = False
	if params.REC == 1:
		REC = True
	else:
		REC = False


	timestep		= 0
	grad_clip		= 10.0
	device			= "cuda" if torch.cuda.is_available() else "cpu"
	if device == "cuda":
		torch.cuda.set_device(did)
	else:
		num_writer		= 1
		num_samples		= 3

	writer_all		= SummaryWriter(logdir='./runs/all-'+cwd)
	if VALIDATION:
		valid_writer_all	= SummaryWriter(logdir='./runs/valid-all-'+cwd)

	print (sentence_loss, word_loss, segment_loss)
	net				= SynthesisNetwork(weight_dim=weight_dim, num_layers=num_layers, sentence_loss=sentence_loss, word_loss=word_loss, segment_loss=segment_loss, TYPE_A=TYPE_A, TYPE_B=TYPE_B, TYPE_C=TYPE_C, TYPE_D=TYPE_D, ORIGINAL=ORIGINAL, REC=REC)
	_				= net.to(device)

	for param in net.parameters():
		nn.init.normal_(param, mean=0.0, std=0.075)

	dl				= DataLoader(num_writer=num_writer, num_samples=num_samples, divider=divider, datadir=datadir)

	optimizer		= optim.Adam(net.parameters(), lr=lr)
	step_size		= int(10000 / (num_writer * num_samples))
	scheduler 		= optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.99)

	while True:
		optimizer.zero_grad()
		timestep		   += num_writer * num_samples

		[all_sentence_level_stroke_in, all_sentence_level_stroke_out, all_sentence_level_stroke_length, all_sentence_level_term,
		all_sentence_level_char, all_sentence_level_char_length, all_word_level_stroke_in, all_word_level_stroke_out,
		all_word_level_stroke_length, all_word_level_term, all_word_level_char, all_word_level_char_length,
		all_segment_level_stroke_in, all_segment_level_stroke_out, all_segment_level_stroke_length, all_segment_level_term,
		all_segment_level_char, all_segment_level_char_length] = dl.next_batch(TYPE='TRAIN')

		batch_sentence_level_stroke_in 		= [torch.FloatTensor(a).to(device) for a in all_sentence_level_stroke_in]
		batch_sentence_level_stroke_out 	= [torch.FloatTensor(a).to(device) for a in all_sentence_level_stroke_out]
		batch_sentence_level_stroke_length 	= [torch.LongTensor(a).to(device).unsqueeze(-1) for a in all_sentence_level_stroke_length]
		batch_sentence_level_term 			= [torch.FloatTensor(a).to(device) for a in all_sentence_level_term]
		batch_sentence_level_char 			= [torch.LongTensor(a).to(device) for a in all_sentence_level_char]
		batch_sentence_level_char_length 	= [torch.LongTensor(a).to(device).unsqueeze(-1) for a in all_sentence_level_char_length]
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

		res = net([batch_sentence_level_stroke_in, batch_sentence_level_stroke_out, batch_sentence_level_stroke_length,
				batch_sentence_level_term, batch_sentence_level_char, batch_sentence_level_char_length,
				batch_word_level_stroke_in, batch_word_level_stroke_out, batch_word_level_stroke_length,
				batch_word_level_term, batch_word_level_char, batch_word_level_char_length, batch_segment_level_stroke_in,
				batch_segment_level_stroke_out, batch_segment_level_stroke_length, batch_segment_level_term,
				batch_segment_level_char, batch_segment_level_char_length])

		total_loss, sentence_losses, word_losses, segment_losses = res

		print ("Step :", timestep, "\tLoss :", total_loss.item(), "\tlr :", optimizer.param_groups[0]['lr'])

		writer_all.add_scalar('ALL/total_loss', total_loss, timestep)

		if sentence_loss:
			[total_sentence_loss, mean_sentence_W_consistency_loss, mean_ORIGINAL_sentence_termination_loss, mean_ORIGINAL_sentence_loc_reconstruct_loss, mean_ORIGINAL_sentence_touch_reconstruct_loss, mean_TYPE_A_sentence_termination_loss, mean_TYPE_A_sentence_loc_reconstruct_loss, mean_TYPE_A_sentence_touch_reconstruct_loss, mean_TYPE_B_sentence_termination_loss, mean_TYPE_B_sentence_loc_reconstruct_loss, mean_TYPE_B_sentence_touch_reconstruct_loss, mean_TYPE_A_sentence_WC_reconstruct_loss, mean_TYPE_B_sentence_WC_reconstruct_loss] = sentence_losses

			writer_all.add_scalar('ALL/total_sentence_loss', total_sentence_loss, timestep)
			writer_sentence.add_scalar('Loss/mean_W_consistency_loss', mean_sentence_W_consistency_loss, timestep)
			if ORIGINAL:
				writer_sentence.add_scalar('Loss/mean_ORIGINAL_loss', mean_ORIGINAL_sentence_termination_loss + mean_ORIGINAL_sentence_loc_reconstruct_loss + mean_ORIGINAL_sentence_touch_reconstruct_loss, timestep)
				writer_sentence.add_scalar('Z_LOSS/mean_ORIGINAL_termination_loss', mean_ORIGINAL_sentence_termination_loss, timestep)
				writer_sentence.add_scalar('Loss_Loc/mean_ORIGINAL_loc_reconstruct_loss', mean_ORIGINAL_sentence_loc_reconstruct_loss, timestep)
				writer_sentence.add_scalar('Z_LOSS/mean_ORIGINAL_touch_reconstruct_loss', mean_ORIGINAL_sentence_touch_reconstruct_loss, timestep)
			if TYPE_A:
				writer_sentence.add_scalar('Loss/mean_TYPE_A_loss', mean_TYPE_A_sentence_termination_loss + mean_TYPE_A_sentence_loc_reconstruct_loss + mean_TYPE_A_sentence_touch_reconstruct_loss, timestep)
				writer_sentence.add_scalar('Z_LOSS/mean_TYPE_A_termination_loss', mean_TYPE_A_sentence_termination_loss, timestep)
				writer_sentence.add_scalar('Loss_Loc/mean_TYPE_A_loc_reconstruct_loss', mean_TYPE_A_sentence_loc_reconstruct_loss, timestep)
				writer_sentence.add_scalar('Z_LOSS/mean_TYPE_A_touch_reconstruct_loss', mean_TYPE_A_sentence_touch_reconstruct_loss, timestep)
				writer_sentence.add_scalar('Z_LOSS/mean_TYPE_A_WC_reconstruct_loss', mean_TYPE_A_sentence_WC_reconstruct_loss, timestep)
			if TYPE_B:
				writer_sentence.add_scalar('Loss/mean_TYPE_B_loss', mean_TYPE_B_sentence_termination_loss + mean_TYPE_B_sentence_loc_reconstruct_loss + mean_TYPE_B_sentence_touch_reconstruct_loss, timestep)
				writer_sentence.add_scalar('Z_LOSS/mean_TYPE_B_termination_loss', mean_TYPE_B_sentence_termination_loss, timestep)
				writer_sentence.add_scalar('Loss_Loc/mean_TYPE_B_loc_reconstruct_loss', mean_TYPE_B_sentence_loc_reconstruct_loss, timestep)
				writer_sentence.add_scalar('Z_LOSS/mean_TYPE_B_touch_reconstruct_loss', mean_TYPE_B_sentence_touch_reconstruct_loss, timestep)
				writer_sentence.add_scalar('Z_LOSS/mean_TYPE_B_WC_reconstruct_loss', mean_TYPE_B_sentence_WC_reconstruct_loss, timestep)

		if word_loss:
			[total_word_loss, mean_word_W_consistency_loss, mean_ORIGINAL_word_termination_loss, mean_ORIGINAL_word_loc_reconstruct_loss, mean_ORIGINAL_word_touch_reconstruct_loss, mean_TYPE_A_word_termination_loss, mean_TYPE_A_word_loc_reconstruct_loss, mean_TYPE_A_word_touch_reconstruct_loss, mean_TYPE_B_word_termination_loss, mean_TYPE_B_word_loc_reconstruct_loss, mean_TYPE_B_word_touch_reconstruct_loss, mean_TYPE_C_word_termination_loss, mean_TYPE_C_word_loc_reconstruct_loss, mean_TYPE_C_word_touch_reconstruct_loss, mean_TYPE_D_word_termination_loss, mean_TYPE_D_word_loc_reconstruct_loss, mean_TYPE_D_word_touch_reconstruct_loss, mean_TYPE_A_word_WC_reconstruct_loss, mean_TYPE_B_word_WC_reconstruct_loss, mean_TYPE_C_word_WC_reconstruct_loss, mean_TYPE_D_word_WC_reconstruct_loss] = word_losses
			writer_all.add_scalar('ALL/total_word_loss', total_word_loss, timestep)
			writer_word.add_scalar('Loss/mean_W_consistency_loss', mean_word_W_consistency_loss, timestep)

			if ORIGINAL:
				writer_word.add_scalar('Loss/mean_ORIGINAL_loss', mean_ORIGINAL_word_termination_loss + mean_ORIGINAL_word_loc_reconstruct_loss + mean_ORIGINAL_word_touch_reconstruct_loss, timestep)
				writer_word.add_scalar('Z_LOSS/mean_ORIGINAL_termination_loss', mean_ORIGINAL_word_termination_loss, timestep)
				writer_word.add_scalar('Loss_Loc/mean_ORIGINAL_loc_reconstruct_loss', mean_ORIGINAL_word_loc_reconstruct_loss, timestep)
				writer_word.add_scalar('Z_LOSS/mean_ORIGINAL_touch_reconstruct_loss', mean_ORIGINAL_word_touch_reconstruct_loss, timestep)
			if TYPE_A:
				writer_word.add_scalar('Loss/mean_TYPE_A_loss', mean_TYPE_A_word_termination_loss + mean_TYPE_A_word_loc_reconstruct_loss + mean_TYPE_A_word_touch_reconstruct_loss, timestep)
				writer_word.add_scalar('Z_LOSS/mean_TYPE_A_termination_loss', mean_TYPE_A_word_termination_loss, timestep)
				writer_word.add_scalar('Loss_Loc/mean_TYPE_A_loc_reconstruct_loss', mean_TYPE_A_word_loc_reconstruct_loss, timestep)
				writer_word.add_scalar('Z_LOSS/mean_TYPE_A_touch_reconstruct_loss', mean_TYPE_A_word_touch_reconstruct_loss, timestep)
				writer_word.add_scalar('Z_LOSS/mean_TYPE_A_WC_reconstruct_loss', mean_TYPE_A_word_WC_reconstruct_loss, timestep)
			if TYPE_B:
				writer_word.add_scalar('Loss/mean_TYPE_B_loss', mean_TYPE_B_word_termination_loss + mean_TYPE_B_word_loc_reconstruct_loss + mean_TYPE_B_word_touch_reconstruct_loss, timestep)
				writer_word.add_scalar('Z_LOSS/mean_TYPE_B_termination_loss', mean_TYPE_B_word_termination_loss, timestep)
				writer_word.add_scalar('Loss_Loc/mean_TYPE_B_loc_reconstruct_loss', mean_TYPE_B_word_loc_reconstruct_loss, timestep)
				writer_word.add_scalar('Z_LOSS/mean_TYPE_B_touch_reconstruct_loss', mean_TYPE_B_word_touch_reconstruct_loss, timestep)
				writer_word.add_scalar('Z_LOSS/mean_TYPE_B_WC_reconstruct_loss', mean_TYPE_B_word_WC_reconstruct_loss, timestep)
			if TYPE_C:
				writer_word.add_scalar('Loss/mean_TYPE_C_loss', mean_TYPE_C_word_termination_loss + mean_TYPE_C_word_loc_reconstruct_loss + mean_TYPE_C_word_touch_reconstruct_loss, timestep)
				writer_word.add_scalar('Z_LOSS/mean_TYPE_C_termination_loss', mean_TYPE_C_word_termination_loss, timestep)
				writer_word.add_scalar('Loss_Loc/mean_TYPE_C_loc_reconstruct_loss', mean_TYPE_C_word_loc_reconstruct_loss, timestep)
				writer_word.add_scalar('Z_LOSS/mean_TYPE_C_touch_reconstruct_loss', mean_TYPE_C_word_touch_reconstruct_loss, timestep)
				writer_word.add_scalar('Z_LOSS/mean_TYPE_C_WC_reconstruct_loss', mean_TYPE_C_word_WC_reconstruct_loss, timestep)
			if TYPE_D:
				writer_word.add_scalar('Loss/mean_TYPE_D_loss', mean_TYPE_D_word_termination_loss + mean_TYPE_D_word_loc_reconstruct_loss + mean_TYPE_D_word_touch_reconstruct_loss, timestep)
				writer_word.add_scalar('Z_LOSS/mean_TYPE_D_termination_loss', mean_TYPE_D_word_termination_loss, timestep)
				writer_word.add_scalar('Loss_Loc/mean_TYPE_D_loc_reconstruct_loss', mean_TYPE_D_word_loc_reconstruct_loss, timestep)
				writer_word.add_scalar('Z_LOSS/mean_TYPE_D_touch_reconstruct_loss', mean_TYPE_D_word_touch_reconstruct_loss, timestep)
				writer_word.add_scalar('Z_LOSS/mean_TYPE_D_WC_reconstruct_loss', mean_TYPE_D_word_WC_reconstruct_loss, timestep)

		if segment_loss:
			[total_segment_loss, mean_segment_W_consistency_loss, mean_ORIGINAL_segment_termination_loss, mean_ORIGINAL_segment_loc_reconstruct_loss, mean_ORIGINAL_segment_touch_reconstruct_loss, mean_TYPE_A_segment_termination_loss, mean_TYPE_A_segment_loc_reconstruct_loss, mean_TYPE_A_segment_touch_reconstruct_loss, mean_TYPE_B_segment_termination_loss, mean_TYPE_B_segment_loc_reconstruct_loss, mean_TYPE_B_segment_touch_reconstruct_loss, mean_TYPE_A_segment_WC_reconstruct_loss, mean_TYPE_B_segment_WC_reconstruct_loss] = segment_losses
			writer_all.add_scalar('ALL/total_segment_loss', total_segment_loss, timestep)
			writer_segment.add_scalar('Loss/mean_W_consistency_loss', mean_segment_W_consistency_loss, timestep)
			if ORIGINAL:
				writer_segment.add_scalar('Loss/mean_ORIGINAL_loss', mean_ORIGINAL_segment_termination_loss + mean_ORIGINAL_segment_loc_reconstruct_loss + mean_ORIGINAL_segment_touch_reconstruct_loss, timestep)
				writer_segment.add_scalar('Z_LOSS/mean_ORIGINAL_termination_loss', mean_ORIGINAL_segment_termination_loss, timestep)
				writer_segment.add_scalar('Loss_Loc/mean_ORIGINAL_loc_reconstruct_loss', mean_ORIGINAL_segment_loc_reconstruct_loss, timestep)
				writer_segment.add_scalar('Z_LOSS/mean_ORIGINAL_touch_reconstruct_loss', mean_ORIGINAL_segment_touch_reconstruct_loss, timestep)
			if TYPE_A:
				writer_segment.add_scalar('Loss/mean_TYPE_A_loss', mean_TYPE_A_segment_termination_loss + mean_TYPE_A_segment_loc_reconstruct_loss + mean_TYPE_A_segment_touch_reconstruct_loss, timestep)
				writer_segment.add_scalar('Z_LOSS/mean_TYPE_A_termination_loss', mean_TYPE_A_segment_termination_loss, timestep)
				writer_segment.add_scalar('Loss_Loc/mean_TYPE_A_loc_reconstruct_loss', mean_TYPE_A_segment_loc_reconstruct_loss, timestep)
				writer_segment.add_scalar('Z_LOSS/mean_TYPE_A_touch_reconstruct_loss', mean_TYPE_A_segment_touch_reconstruct_loss, timestep)
				writer_segment.add_scalar('Z_LOSS/mean_TYPE_A_WC_reconstruct_loss', mean_TYPE_A_segment_WC_reconstruct_loss, timestep)
			if TYPE_B:
				writer_segment.add_scalar('Loss/mean_TYPE_B_loss', mean_TYPE_B_segment_termination_loss + mean_TYPE_B_segment_loc_reconstruct_loss + mean_TYPE_B_segment_touch_reconstruct_loss, timestep)
				writer_segment.add_scalar('Z_LOSS/mean_TYPE_B_termination_loss', mean_TYPE_B_segment_termination_loss, timestep)
				writer_segment.add_scalar('Loss_Loc/mean_TYPE_B_loc_reconstruct_loss', mean_TYPE_B_segment_loc_reconstruct_loss, timestep)
				writer_segment.add_scalar('Z_LOSS/mean_TYPE_B_touch_reconstruct_loss', mean_TYPE_B_segment_touch_reconstruct_loss, timestep)
				writer_segment.add_scalar('Z_LOSS/mean_TYPE_B_WC_reconstruct_loss', mean_TYPE_B_segment_WC_reconstruct_loss, timestep)

		total_loss.backward()

		torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
		for p in net.parameters():
			if p.grad is not None:
				p.data.add_(-lr, p.grad.data)

		optimizer.step()

		if timestep % (num_writer * num_samples * 1) == 0.0:
			commands_list = net.sample([	batch_word_level_stroke_in, batch_word_level_stroke_out, batch_word_level_stroke_length,
											batch_word_level_term, batch_word_level_char, batch_word_level_char_length, batch_segment_level_stroke_in,
											batch_segment_level_stroke_out, batch_segment_level_stroke_length, batch_segment_level_term,
											batch_segment_level_char, batch_segment_level_char_length])
			[t_commands, o_commands, a_commands, b_commands, c_commands, d_commands] = commands_list

			t_im = Image.fromarray(np.zeros([160, 750]))
			t_dr = ImageDraw.Draw(t_im)

			px, py = 30, 100
			for i, [dx,dy,t] in enumerate(t_commands):
				x = px + dx * 5
				y = py + dy * 5
				if t == 0:
					t_dr.line((px,py,x,y),255,1)
				px, py = x, y

			o_im = Image.fromarray(np.zeros([160, 750]))
			o_dr = ImageDraw.Draw(o_im)
			px, py = 30, 100
			for i, [dx,dy,t] in enumerate(o_commands):
				x = px + dx * 5
				y = py + dy * 5
				if t == 0:
					o_dr.line((px,py,x,y),255,1)
				px, py = x, y

			a_im = Image.fromarray(np.zeros([160, 750]))
			a_dr = ImageDraw.Draw(a_im)
			px, py = 30, 100
			for i, [dx,dy,t] in enumerate(a_commands):
				x = px + dx * 5
				y = py + dy * 5
				if t == 0:
					a_dr.line((px,py,x,y),255,1)
				px, py = x, y

			b_im = Image.fromarray(np.zeros([160, 750]))
			b_dr = ImageDraw.Draw(b_im)
			px, py = 30, 100
			for i, [dx,dy,t] in enumerate(b_commands):
				x = px + dx * 5
				y = py + dy * 5
				if t == 0:
					b_dr.line((px,py,x,y),255,1)
				px, py = x, y

			c_im = Image.fromarray(np.zeros([160, 750]))
			c_dr = ImageDraw.Draw(c_im)
			px, py = 30, 100
			for i, [dx,dy,t] in enumerate(c_commands):
				x = px + dx * 5
				y = py + dy * 5
				if t == 0:
					c_dr.line((px,py,x,y),255,1)
				px, py = x, y

			d_im = Image.fromarray(np.zeros([160, 750]))
			d_dr = ImageDraw.Draw(d_im)
			px, py = 30, 100
			for i, [dx,dy,t] in enumerate(d_commands):
				x = px + dx * 5
				y = py + dy * 5
				if t == 0:
					d_dr.line((px,py,x,y),255,1)
				px, py = x, y

			dst = Image.new('RGB', (750, 960))
			dst.paste(t_im, (0, 0))
			dst.paste(o_im, (0, 160))
			dst.paste(a_im, (0, 320))
			dst.paste(b_im, (0, 480))
			dst.paste(c_im, (0, 640))
			dst.paste(d_im, (0, 800))
			writer_all.add_image('Res/Results', np.asarray(dst.convert("RGB")), timestep, dataformats='HWC')

		if VALIDATION:
			[all_sentence_level_stroke_in, all_sentence_level_stroke_out, all_sentence_level_stroke_length, all_sentence_level_term,
			all_sentence_level_char, all_sentence_level_char_length, all_word_level_stroke_in, all_word_level_stroke_out,
			all_word_level_stroke_length, all_word_level_term, all_word_level_char, all_word_level_char_length,
			all_segment_level_stroke_in, all_segment_level_stroke_out, all_segment_level_stroke_length, all_segment_level_term,
			all_segment_level_char, all_segment_level_char_length] = dl.next_batch(TYPE='VALID')

			batch_sentence_level_stroke_in 		= [torch.FloatTensor(a).to(device) for a in all_sentence_level_stroke_in]
			batch_sentence_level_stroke_out 	= [torch.FloatTensor(a).to(device) for a in all_sentence_level_stroke_out]
			batch_sentence_level_stroke_length 	= [torch.LongTensor(a).to(device).unsqueeze(-1) for a in all_sentence_level_stroke_length]
			batch_sentence_level_term 			= [torch.FloatTensor(a).to(device) for a in all_sentence_level_term]
			batch_sentence_level_char 			= [torch.LongTensor(a).to(device) for a in all_sentence_level_char]
			batch_sentence_level_char_length 	= [torch.LongTensor(a).to(device).unsqueeze(-1) for a in all_sentence_level_char_length]
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

			res = net([batch_sentence_level_stroke_in, batch_sentence_level_stroke_out, batch_sentence_level_stroke_length,
					batch_sentence_level_term, batch_sentence_level_char, batch_sentence_level_char_length,
					batch_word_level_stroke_in, batch_word_level_stroke_out, batch_word_level_stroke_length,
					batch_word_level_term, batch_word_level_char, batch_word_level_char_length, batch_segment_level_stroke_in,
					batch_segment_level_stroke_out, batch_segment_level_stroke_length, batch_segment_level_term,
					batch_segment_level_char, batch_segment_level_char_length])

			total_loss, sentence_losses, word_losses, segment_losses = res

			valid_writer_all.add_scalar('ALL/total_loss', total_loss, timestep)

			if sentence_loss:
				[total_sentence_loss, mean_sentence_W_consistency_loss, mean_ORIGINAL_sentence_termination_loss, mean_ORIGINAL_sentence_loc_reconstruct_loss, mean_ORIGINAL_sentence_touch_reconstruct_loss, mean_TYPE_A_sentence_termination_loss, mean_TYPE_A_sentence_loc_reconstruct_loss, mean_TYPE_A_sentence_touch_reconstruct_loss, mean_TYPE_B_sentence_termination_loss, mean_TYPE_B_sentence_loc_reconstruct_loss, mean_TYPE_B_sentence_touch_reconstruct_loss, mean_TYPE_A_sentence_WC_reconstruct_loss, mean_TYPE_B_sentence_WC_reconstruct_loss] = sentence_losses

				valid_writer_all.add_scalar('ALL/total_sentence_loss', total_sentence_loss, timestep)
				valid_writer_sentence.add_scalar('Loss/mean_W_consistency_loss', mean_sentence_W_consistency_loss, timestep)
				if ORIGINAL:
					valid_writer_sentence.add_scalar('Loss/mean_ORIGINAL_loss', mean_ORIGINAL_sentence_termination_loss + mean_ORIGINAL_sentence_loc_reconstruct_loss + mean_ORIGINAL_sentence_touch_reconstruct_loss, timestep)
					valid_writer_sentence.add_scalar('Z_LOSS/mean_ORIGINAL_termination_loss', mean_ORIGINAL_sentence_termination_loss, timestep)
					valid_writer_sentence.add_scalar('Loss_Loc/mean_ORIGINAL_loc_reconstruct_loss', mean_ORIGINAL_sentence_loc_reconstruct_loss, timestep)
					valid_writer_sentence.add_scalar('Z_LOSS/mean_ORIGINAL_touch_reconstruct_loss', mean_ORIGINAL_sentence_touch_reconstruct_loss, timestep)
				if TYPE_A:
					valid_writer_sentence.add_scalar('Loss/mean_TYPE_A_loss', mean_TYPE_A_sentence_termination_loss + mean_TYPE_A_sentence_loc_reconstruct_loss + mean_TYPE_A_sentence_touch_reconstruct_loss, timestep)
					valid_writer_sentence.add_scalar('Z_LOSS/mean_TYPE_A_termination_loss', mean_TYPE_A_sentence_termination_loss, timestep)
					valid_writer_sentence.add_scalar('Loss_Loc/mean_TYPE_A_loc_reconstruct_loss', mean_TYPE_A_sentence_loc_reconstruct_loss, timestep)
					valid_writer_sentence.add_scalar('Z_LOSS/mean_TYPE_A_touch_reconstruct_loss', mean_TYPE_A_sentence_touch_reconstruct_loss, timestep)
					valid_writer_sentence.add_scalar('Z_LOSS/mean_TYPE_A_WC_reconstruct_loss', mean_TYPE_A_sentence_WC_reconstruct_loss, timestep)
				if TYPE_B:
					valid_writer_sentence.add_scalar('Loss/mean_TYPE_B_loss', mean_TYPE_B_sentence_termination_loss + mean_TYPE_B_sentence_loc_reconstruct_loss + mean_TYPE_B_sentence_touch_reconstruct_loss, timestep)
					valid_writer_sentence.add_scalar('Z_LOSS/mean_TYPE_B_termination_loss', mean_TYPE_B_sentence_termination_loss, timestep)
					valid_writer_sentence.add_scalar('Loss_Loc/mean_TYPE_B_loc_reconstruct_loss', mean_TYPE_B_sentence_loc_reconstruct_loss, timestep)
					valid_writer_sentence.add_scalar('Z_LOSS/mean_TYPE_B_touch_reconstruct_loss', mean_TYPE_B_sentence_touch_reconstruct_loss, timestep)
					valid_writer_sentence.add_scalar('Z_LOSS/mean_TYPE_B_WC_reconstruct_loss', mean_TYPE_B_sentence_WC_reconstruct_loss, timestep)

			if word_loss:
				[total_word_loss, mean_word_W_consistency_loss, mean_ORIGINAL_word_termination_loss, mean_ORIGINAL_word_loc_reconstruct_loss, mean_ORIGINAL_word_touch_reconstruct_loss, mean_TYPE_A_word_termination_loss, mean_TYPE_A_word_loc_reconstruct_loss, mean_TYPE_A_word_touch_reconstruct_loss, mean_TYPE_B_word_termination_loss, mean_TYPE_B_word_loc_reconstruct_loss, mean_TYPE_B_word_touch_reconstruct_loss, mean_TYPE_C_word_termination_loss, mean_TYPE_C_word_loc_reconstruct_loss, mean_TYPE_C_word_touch_reconstruct_loss, mean_TYPE_D_word_termination_loss, mean_TYPE_D_word_loc_reconstruct_loss, mean_TYPE_D_word_touch_reconstruct_loss, mean_TYPE_A_word_WC_reconstruct_loss, mean_TYPE_B_word_WC_reconstruct_loss, mean_TYPE_C_word_WC_reconstruct_loss, mean_TYPE_D_word_WC_reconstruct_loss] = word_losses
				valid_writer_all.add_scalar('ALL/total_word_loss', total_word_loss, timestep)
				valid_writer_word.add_scalar('Loss/mean_W_consistency_loss', mean_word_W_consistency_loss, timestep)

				if ORIGINAL:
					valid_writer_word.add_scalar('Loss/mean_ORIGINAL_loss', mean_ORIGINAL_word_termination_loss + mean_ORIGINAL_word_loc_reconstruct_loss + mean_ORIGINAL_word_touch_reconstruct_loss, timestep)
					valid_writer_word.add_scalar('Z_LOSS/mean_ORIGINAL_termination_loss', mean_ORIGINAL_word_termination_loss, timestep)
					valid_writer_word.add_scalar('Loss_Loc/mean_ORIGINAL_loc_reconstruct_loss', mean_ORIGINAL_word_loc_reconstruct_loss, timestep)
					valid_writer_word.add_scalar('Z_LOSS/mean_ORIGINAL_touch_reconstruct_loss', mean_ORIGINAL_word_touch_reconstruct_loss, timestep)
				if TYPE_A:
					valid_writer_word.add_scalar('Loss/mean_TYPE_A_loss', mean_TYPE_A_word_termination_loss + mean_TYPE_A_word_loc_reconstruct_loss + mean_TYPE_A_word_touch_reconstruct_loss, timestep)
					valid_writer_word.add_scalar('Z_LOSS/mean_TYPE_A_termination_loss', mean_TYPE_A_word_termination_loss, timestep)
					valid_writer_word.add_scalar('Loss_Loc/mean_TYPE_A_loc_reconstruct_loss', mean_TYPE_A_word_loc_reconstruct_loss, timestep)
					valid_writer_word.add_scalar('Z_LOSS/mean_TYPE_A_touch_reconstruct_loss', mean_TYPE_A_word_touch_reconstruct_loss, timestep)
					valid_writer_word.add_scalar('Z_LOSS/mean_TYPE_A_WC_reconstruct_loss', mean_TYPE_A_word_WC_reconstruct_loss, timestep)
				if TYPE_B:
					valid_writer_word.add_scalar('Loss/mean_TYPE_B_loss', mean_TYPE_B_word_termination_loss + mean_TYPE_B_word_loc_reconstruct_loss + mean_TYPE_B_word_touch_reconstruct_loss, timestep)
					valid_writer_word.add_scalar('Z_LOSS/mean_TYPE_B_termination_loss', mean_TYPE_B_word_termination_loss, timestep)
					valid_writer_word.add_scalar('Loss_Loc/mean_TYPE_B_loc_reconstruct_loss', mean_TYPE_B_word_loc_reconstruct_loss, timestep)
					valid_writer_word.add_scalar('Z_LOSS/mean_TYPE_B_touch_reconstruct_loss', mean_TYPE_B_word_touch_reconstruct_loss, timestep)
					valid_writer_word.add_scalar('Z_LOSS/mean_TYPE_B_WC_reconstruct_loss', mean_TYPE_B_word_WC_reconstruct_loss, timestep)
				if TYPE_C:
					valid_writer_word.add_scalar('Loss/mean_TYPE_C_loss', mean_TYPE_C_word_termination_loss + mean_TYPE_C_word_loc_reconstruct_loss + mean_TYPE_C_word_touch_reconstruct_loss, timestep)
					valid_writer_word.add_scalar('Z_LOSS/mean_TYPE_C_termination_loss', mean_TYPE_C_word_termination_loss, timestep)
					valid_writer_word.add_scalar('Loss_Loc/mean_TYPE_C_loc_reconstruct_loss', mean_TYPE_C_word_loc_reconstruct_loss, timestep)
					valid_writer_word.add_scalar('Z_LOSS/mean_TYPE_C_touch_reconstruct_loss', mean_TYPE_C_word_touch_reconstruct_loss, timestep)
					valid_writer_word.add_scalar('Z_LOSS/mean_TYPE_C_WC_reconstruct_loss', mean_TYPE_C_word_WC_reconstruct_loss, timestep)
				if TYPE_D:
					valid_writer_word.add_scalar('Loss/mean_TYPE_D_loss', mean_TYPE_D_word_termination_loss + mean_TYPE_D_word_loc_reconstruct_loss + mean_TYPE_D_word_touch_reconstruct_loss, timestep)
					valid_writer_word.add_scalar('Z_LOSS/mean_TYPE_D_termination_loss', mean_TYPE_D_word_termination_loss, timestep)
					valid_writer_word.add_scalar('Loss_Loc/mean_TYPE_D_loc_reconstruct_loss', mean_TYPE_D_word_loc_reconstruct_loss, timestep)
					valid_writer_word.add_scalar('Z_LOSS/mean_TYPE_D_touch_reconstruct_loss', mean_TYPE_D_word_touch_reconstruct_loss, timestep)
					valid_writer_word.add_scalar('Z_LOSS/mean_TYPE_D_WC_reconstruct_loss', mean_TYPE_D_word_WC_reconstruct_loss, timestep)

			if segment_loss:
				[total_segment_loss, mean_segment_W_consistency_loss, mean_ORIGINAL_segment_termination_loss, mean_ORIGINAL_segment_loc_reconstruct_loss, mean_ORIGINAL_segment_touch_reconstruct_loss, mean_TYPE_A_segment_termination_loss, mean_TYPE_A_segment_loc_reconstruct_loss, mean_TYPE_A_segment_touch_reconstruct_loss, mean_TYPE_B_segment_termination_loss, mean_TYPE_B_segment_loc_reconstruct_loss, mean_TYPE_B_segment_touch_reconstruct_loss, mean_TYPE_A_segment_WC_reconstruct_loss, mean_TYPE_B_segment_WC_reconstruct_loss] = segment_losses
				valid_writer_all.add_scalar('ALL/total_segment_loss', total_segment_loss, timestep)
				valid_writer_segment.add_scalar('Loss/mean_W_consistency_loss', mean_segment_W_consistency_loss, timestep)
				if ORIGINAL:
					valid_writer_segment.add_scalar('Loss/mean_ORIGINAL_loss', mean_ORIGINAL_segment_termination_loss + mean_ORIGINAL_segment_loc_reconstruct_loss + mean_ORIGINAL_segment_touch_reconstruct_loss, timestep)
					valid_writer_segment.add_scalar('Z_LOSS/mean_ORIGINAL_termination_loss', mean_ORIGINAL_segment_termination_loss, timestep)
					valid_writer_segment.add_scalar('Loss_Loc/mean_ORIGINAL_loc_reconstruct_loss', mean_ORIGINAL_segment_loc_reconstruct_loss, timestep)
					valid_writer_segment.add_scalar('Z_LOSS/mean_ORIGINAL_touch_reconstruct_loss', mean_ORIGINAL_segment_touch_reconstruct_loss, timestep)
				if TYPE_A:
					valid_writer_segment.add_scalar('Loss/mean_TYPE_A_loss', mean_TYPE_A_segment_termination_loss + mean_TYPE_A_segment_loc_reconstruct_loss + mean_TYPE_A_segment_touch_reconstruct_loss, timestep)
					valid_writer_segment.add_scalar('Z_LOSS/mean_TYPE_A_termination_loss', mean_TYPE_A_segment_termination_loss, timestep)
					valid_writer_segment.add_scalar('Loss_Loc/mean_TYPE_A_loc_reconstruct_loss', mean_TYPE_A_segment_loc_reconstruct_loss, timestep)
					valid_writer_segment.add_scalar('Z_LOSS/mean_TYPE_A_touch_reconstruct_loss', mean_TYPE_A_segment_touch_reconstruct_loss, timestep)
					valid_writer_segment.add_scalar('Z_LOSS/mean_TYPE_A_WC_reconstruct_loss', mean_TYPE_A_segment_WC_reconstruct_loss, timestep)
				if TYPE_B:
					valid_writer_segment.add_scalar('Loss/mean_TYPE_B_loss', mean_TYPE_B_segment_termination_loss + mean_TYPE_B_segment_loc_reconstruct_loss + mean_TYPE_B_segment_touch_reconstruct_loss, timestep)
					valid_writer_segment.add_scalar('Z_LOSS/mean_TYPE_B_termination_loss', mean_TYPE_B_segment_termination_loss, timestep)
					valid_writer_segment.add_scalar('Loss_Loc/mean_TYPE_B_loc_reconstruct_loss', mean_TYPE_B_segment_loc_reconstruct_loss, timestep)
					valid_writer_segment.add_scalar('Z_LOSS/mean_TYPE_B_touch_reconstruct_loss', mean_TYPE_B_segment_touch_reconstruct_loss, timestep)
					valid_writer_segment.add_scalar('Z_LOSS/mean_TYPE_B_WC_reconstruct_loss', mean_TYPE_B_segment_WC_reconstruct_loss, timestep)

		if timestep % (num_writer * num_samples * 1000) == 0.0:
			torch.save(net.state_dict(), 'model/'+str(timestep)+'.pt')

	writer.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Arguments for training the handwriting synthesis network.')

	parser.add_argument('--divider', type=float, default=5.0)
	parser.add_argument('--weight_dim', type=int, default=256)
	parser.add_argument('--sample', type=int, default=2)
	parser.add_argument('--device', type=int, default=1)
	parser.add_argument('--num_layers', type=int, default=3)
	parser.add_argument('--num_writer', type=int, default=1)
	parser.add_argument('--lr', type=float, default=0.001)

	parser.add_argument('--sentence_loss', type=int, default=1)
	parser.add_argument('--word_loss', type=int, default=1)
	parser.add_argument('--segment_loss', type=int, default=1)

	parser.add_argument('--TYPE_A', type=int, default=1)
	parser.add_argument('--TYPE_B', type=int, default=1)
	parser.add_argument('--TYPE_C', type=int, default=1)
	parser.add_argument('--TYPE_D', type=int, default=1)
	parser.add_argument('--ORIGINAL', type=int, default=1)

	parser.add_argument('--VALIDATION', type=int, default=1)
	parser.add_argument('--no_char', type=int, default=0)
	parser.add_argument('--REC', type=int, default=1)

	main(parser.parse_args())

