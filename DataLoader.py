import os
import numpy as np
import torch
import random
from PIL import Image, ImageDraw, ImageFont
import pickle
from config.GlobalVariables import *

np.random.seed(0)

class DataLoader():
	def __init__(self, num_writer=2, num_samples=5, divider=10.0, datadir='./data/writers'):
		self.device			= 'cuda' if torch.cuda.is_available() else 'cpu'
		self.num_writer		= num_writer
		self.num_samples	= num_samples
		self.divider		= divider
		self.datadir		= datadir
		print ('self.datadir : ', self.datadir)
		self.total_writers	= len([name for name in os.listdir(datadir)])

	def next_batch(self, TYPE='TRAIN', uid=-1, tids=[]):
		all_sentence_level_stroke_in		= []
		all_sentence_level_stroke_out		= []
		all_sentence_level_stroke_length	= []
		all_sentence_level_term				= []
		all_sentence_level_char				= []
		all_sentence_level_char_length		= []
		all_word_level_stroke_in			= []
		all_word_level_stroke_out			= []
		all_word_level_stroke_length		= []
		all_word_level_term					= []
		all_word_level_char					= []
		all_word_level_char_length			= []
		all_segment_level_stroke_in			= []
		all_segment_level_stroke_out		= []
		all_segment_level_stroke_length		= []
		all_segment_level_term				= []
		all_segment_level_char				= []
		all_segment_level_char_length		= []

		while len(all_sentence_level_stroke_in) < self.num_writer:
			if uid < 0:
				if TYPE == 'TRAIN':
					if self.datadir == './data/NEW_writers' or self.datadir == './data/writers':
						uid = np.random.choice([i for i in range(150)])
					else:
						if self.device == 'cpu':
							uid = np.random.choice([i for i in range(20)])
						else:
							uid = np.random.choice([i for i in range(294)])
				else:
					uid = np.random.choice([i for i in range(150,170)])

			total_texts				= len([name for name in os.listdir(self.datadir+'/'+str(uid))])
			if len(tids) == 0:
				tids = random.sample([i for i in range(total_texts)], self.num_samples)

			user_sentence_level_stroke_in		= []
			user_sentence_level_stroke_out		= []
			user_sentence_level_stroke_length	= []
			user_sentence_level_term			= []
			user_sentence_level_char			= []
			user_sentence_level_char_length		= []
			user_word_level_stroke_in			= []
			user_word_level_stroke_out			= []
			user_word_level_stroke_length		= []
			user_word_level_term				= []
			user_word_level_char				= []
			user_word_level_char_length			= []
			user_segment_level_stroke_in		= []
			user_segment_level_stroke_out		= []
			user_segment_level_stroke_length	= []
			user_segment_level_term				= []
			user_segment_level_char				= []
			user_segment_level_char_length		= []

			# print ("uid: ", uid, "\ttids:", tids)
			for tid in tids:
				if self.datadir == './data/NEW_writers':
					[sentence_level_raw_stroke, sentence_level_stroke_in, sentence_level_stroke_out, sentence_level_term, sentence_level_char, word_level_raw_stroke, word_level_stroke_in, word_level_stroke_out, word_level_term, word_level_char, segment_level_raw_stroke, segment_level_stroke_in, segment_level_stroke_out, segment_level_term, segment_level_char] = \
						np.load(self.datadir+'/'+str(uid)+'/'+str(tid)+'.npy', allow_pickle=True, encoding='bytes')

				elif self.datadir == './data/DW_writers':
					[sentence_level_raw_stroke, sentence_level_char, sentence_level_term, sentence_level_stroke_in, sentence_level_stroke_out,
					word_level_raw_stroke, word_level_char, word_level_term, word_level_stroke_in, word_level_stroke_out,
					segment_level_raw_stroke, segment_level_char, segment_level_term, segment_level_stroke_in, segment_level_stroke_out, _] = \
						np.load(self.datadir+'/'+str(uid)+'/'+str(tid)+'.npy', allow_pickle=True, encoding='bytes')

				elif self.datadir == './data/VALID_DW_writers':
					[sentence_level_raw_stroke, sentence_level_char, sentence_level_term, sentence_level_stroke_in, sentence_level_stroke_out,
					word_level_raw_stroke, word_level_char, word_level_term, word_level_stroke_in, word_level_stroke_out,
					segment_level_raw_stroke, segment_level_char, segment_level_term, segment_level_stroke_in, segment_level_stroke_out, _] = \
						np.load(self.datadir+'/'+str(uid)+'/'+str(tid)+'.npy', allow_pickle=True, encoding='bytes')

				else:
					[sentence_level_raw_stroke, sentence_level_stroke_in, sentence_level_stroke_out, sentence_level_term, sentence_level_char, word_level_raw_stroke, word_level_stroke_in, word_level_stroke_out, word_level_term, word_level_char, segment_level_raw_stroke, segment_level_stroke_in, segment_level_stroke_out, segment_level_term, segment_level_char, _] = \
						np.load(self.datadir+'/'+str(uid)+'/'+str(tid)+'.npy', allow_pickle=True, encoding='bytes')

				if self.datadir == './data/DW_writers':
					sentence_level_char	= sentence_level_char[1:]
					sentence_level_term	= sentence_level_term[1:]

				if self.datadir == './data/VALID_DW_writers':
					sentence_level_char	= sentence_level_char[1:]
					sentence_level_term	= sentence_level_term[1:]

				while True:
					if len(sentence_level_term) == 0:
						break
					if sentence_level_term[-1] != 1.0:
						sentence_level_raw_stroke = sentence_level_raw_stroke[:-1]
						sentence_level_char = sentence_level_char[:-1]
						sentence_level_term = sentence_level_term[:-1]
						sentence_level_stroke_in = sentence_level_stroke_in[:-1]
						sentence_level_stroke_out = sentence_level_stroke_out[:-1]
					else:
						break

				tmp = []
				for i, t in enumerate(sentence_level_term):
					if t == 1:
						tmp.append(sentence_level_char[i])

				a = np.ones_like(sentence_level_stroke_in)
				a[:,:2] /= self.divider

				if len(sentence_level_stroke_in) == len(sentence_level_term) and len(tmp) > 0 and len(sentence_level_stroke_in) > 0:
					user_sentence_level_stroke_in.append(np.asarray(sentence_level_stroke_in) * a)
					user_sentence_level_stroke_out.append(np.asarray(sentence_level_stroke_out) * a)
					user_sentence_level_stroke_length.append(len(sentence_level_stroke_in))
					user_sentence_level_char.append(np.asarray(tmp))
					user_sentence_level_term.append(np.asarray(sentence_level_term))
					user_sentence_level_char_length.append(len(tmp))

				for wid in range(len(word_level_stroke_in)):
					each_word_level_stroke_in		= word_level_stroke_in[wid]
					each_word_level_stroke_out		= word_level_stroke_out[wid]

					if self.datadir == './data/DW_writers':
						each_word_level_term			= word_level_term[wid][1:]
						each_word_level_char			= word_level_char[wid][1:]
					elif self.datadir == './data/VALID_DW_writers':
						each_word_level_term			= word_level_term[wid][1:]
						each_word_level_char			= word_level_char[wid][1:]
					else:
						each_word_level_term			= word_level_term[wid]
						each_word_level_char			= word_level_char[wid]


					# assert (len(each_word_level_stroke_in) == len(each_word_level_char) == len(each_word_level_term))
					while True:
						if len(each_word_level_term) == 0:
							break
						if each_word_level_term[-1] != 1.0:
							# each_word_level_raw_stroke = each_word_level_raw_stroke[:-1]
							each_word_level_char = each_word_level_char[:-1]
							each_word_level_term = each_word_level_term[:-1]
							each_word_level_stroke_in = each_word_level_stroke_in[:-1]
							each_word_level_stroke_out = each_word_level_stroke_out[:-1]
						else:
							break

					tmp = []
					for i, t in enumerate(each_word_level_term):
						if t == 1:
							tmp.append(each_word_level_char[i])

					b = np.ones_like(each_word_level_stroke_in)
					b[:,:2] /= self.divider

					if len(each_word_level_stroke_in) == len(each_word_level_term) and len(tmp) > 0 and len(each_word_level_stroke_in) > 0:
						user_word_level_stroke_in.append(np.asarray(each_word_level_stroke_in) * b)
						user_word_level_stroke_out.append(np.asarray(each_word_level_stroke_out) * b)
						user_word_level_stroke_length.append(len(each_word_level_stroke_in))
						user_word_level_char.append(np.asarray(tmp))
						user_word_level_term.append(np.asarray(each_word_level_term))
						user_word_level_char_length.append(len(tmp))

					segment_level_stroke_in_list		= []
					segment_level_stroke_out_list		= []
					segment_level_stroke_length_list	= []
					segment_level_char_list				= []
					segment_level_term_list				= []
					segment_level_char_length_list		= []

					for sid in range(len(segment_level_stroke_in[wid])):
						each_segment_level_stroke_in	= segment_level_stroke_in[wid][sid]
						each_segment_level_stroke_out	= segment_level_stroke_out[wid][sid]

						if self.datadir == './data/DW_writers':
							each_segment_level_term			= segment_level_term[wid][sid][1:]
							each_segment_level_char			= segment_level_char[wid][sid][1:]
						elif self.datadir == './data/VALID_DW_writers':
							each_segment_level_term			= segment_level_term[wid][sid][1:]
							each_segment_level_char			= segment_level_char[wid][sid][1:]
						else:
							each_segment_level_term			= segment_level_term[wid][sid]
							each_segment_level_char			= segment_level_char[wid][sid]

						while True:
							if len(each_segment_level_term) == 0:
								break
							if each_segment_level_term[-1] != 1.0:
								# each_segment_level_raw_stroke = each_segment_level_raw_stroke[:-1]
								each_segment_level_char = each_segment_level_char[:-1]
								each_segment_level_term = each_segment_level_term[:-1]
								each_segment_level_stroke_in = each_segment_level_stroke_in[:-1]
								each_segment_level_stroke_out = each_segment_level_stroke_out[:-1]
							else:
								break

						tmp = []
						for i, t in enumerate(each_segment_level_term):
							if t == 1:
								tmp.append(each_segment_level_char[i])

						c = np.ones_like(each_segment_level_stroke_in)
						c[:,:2] /= self.divider

						if len(each_segment_level_stroke_in) == len(each_segment_level_term) and len(tmp) > 0 and len(each_segment_level_stroke_in) > 0:
							segment_level_stroke_in_list.append(np.asarray(each_segment_level_stroke_in) * c)
							segment_level_stroke_out_list.append(np.asarray(each_segment_level_stroke_out) * c)
							segment_level_stroke_length_list.append(len(each_segment_level_stroke_in))
							segment_level_char_list.append(np.asarray(tmp))
							segment_level_term_list.append(np.asarray(each_segment_level_term))
							segment_level_char_length_list.append(len(tmp))

					if len(segment_level_stroke_length_list) > 0:
						SEGMENT_MAX_STROKE_LENGTH		= np.max(segment_level_stroke_length_list)
						SEGMENT_MAX_CHARACTER_LENGTH	= np.max(segment_level_char_length_list)

						new_segment_level_stroke_in_list 	= np.asarray([np.pad(a, ((0, SEGMENT_MAX_STROKE_LENGTH-len(a)), (0, 0)), 'constant') for a in segment_level_stroke_in_list])
						new_segment_level_stroke_out_list 	= np.asarray([np.pad(a, ((0, SEGMENT_MAX_STROKE_LENGTH-len(a)), (0, 0)), 'constant') for a in segment_level_stroke_out_list])
						new_segment_level_term_list 		= np.asarray([np.pad(a, ((0, SEGMENT_MAX_STROKE_LENGTH-len(a))), 'constant') for a in segment_level_term_list])
						new_segment_level_char_list 		= np.asarray([np.pad(a, ((0, SEGMENT_MAX_CHARACTER_LENGTH-len(a))), 'constant') for a in segment_level_char_list])

						user_segment_level_stroke_in.append(new_segment_level_stroke_in_list)
						user_segment_level_stroke_out.append(new_segment_level_stroke_out_list)
						user_segment_level_stroke_length.append(segment_level_stroke_length_list)
						user_segment_level_char.append(new_segment_level_char_list)
						user_segment_level_term.append(new_segment_level_term_list)
						user_segment_level_char_length.append(segment_level_char_length_list)

			WORD_MAX_STROKE_LENGTH			= np.max(user_word_level_stroke_length)
			WORD_MAX_CHARACTER_LENGTH		= np.max(user_word_level_char_length)

			SENTENCE_MAX_STROKE_LENGTH		= np.max(user_sentence_level_stroke_length)
			SENTENCE_MAX_CHARACTER_LENGTH	= np.max(user_sentence_level_char_length)

			new_sentence_level_stroke_in	= np.asarray([np.pad(a, ((0, SENTENCE_MAX_STROKE_LENGTH-len(a)), (0,0)), 'constant') for a in user_sentence_level_stroke_in])
			new_sentence_level_stroke_out	= np.asarray([np.pad(a, ((0, SENTENCE_MAX_STROKE_LENGTH-len(a)), (0,0)), 'constant') for a in user_sentence_level_stroke_out])
			new_sentence_level_term			= np.asarray([np.pad(a, ((0, SENTENCE_MAX_STROKE_LENGTH-len(a))), 'constant') for a in user_sentence_level_term])
			new_sentence_level_char			= np.asarray([np.pad(a, ((0, SENTENCE_MAX_CHARACTER_LENGTH-len(a))), 'constant') for a in user_sentence_level_char])
			new_word_level_stroke_in		= np.asarray([np.pad(a, ((0, WORD_MAX_STROKE_LENGTH-len(a)), (0,0)), 'constant') for a in user_word_level_stroke_in])
			new_word_level_stroke_out		= np.asarray([np.pad(a, ((0, WORD_MAX_STROKE_LENGTH-len(a)), (0,0)), 'constant') for a in user_word_level_stroke_out])
			new_word_level_term				= np.asarray([np.pad(a, ((0, WORD_MAX_STROKE_LENGTH-len(a))), 'constant') for a in user_word_level_term])
			new_word_level_char				= np.asarray([np.pad(a, ((0, WORD_MAX_CHARACTER_LENGTH-len(a))), 'constant') for a in user_word_level_char])

			all_sentence_level_stroke_in.append(new_sentence_level_stroke_in)
			all_sentence_level_stroke_out.append(new_sentence_level_stroke_out)
			all_sentence_level_stroke_length.append(user_sentence_level_stroke_length)
			all_sentence_level_term.append(new_sentence_level_term)
			all_sentence_level_char.append(new_sentence_level_char)
			all_sentence_level_char_length.append(user_sentence_level_char_length)
			all_word_level_stroke_in.append(new_word_level_stroke_in)
			all_word_level_stroke_out.append(new_word_level_stroke_out)
			all_word_level_stroke_length.append(user_word_level_stroke_length)
			all_word_level_term.append(new_word_level_term)
			all_word_level_char.append(new_word_level_char)
			all_word_level_char_length.append(user_word_level_char_length)
			all_segment_level_stroke_in.append(user_segment_level_stroke_in)
			all_segment_level_stroke_out.append(user_segment_level_stroke_out)
			all_segment_level_stroke_length.append(user_segment_level_stroke_length)
			all_segment_level_term.append(user_segment_level_term)
			all_segment_level_char.append(user_segment_level_char)
			all_segment_level_char_length.append(user_segment_level_char_length)

		return [all_sentence_level_stroke_in, all_sentence_level_stroke_out, all_sentence_level_stroke_length, all_sentence_level_term, all_sentence_level_char, all_sentence_level_char_length, all_word_level_stroke_in, all_word_level_stroke_out, all_word_level_stroke_length, all_word_level_term, all_word_level_char, all_word_level_char_length, all_segment_level_stroke_in, all_segment_level_stroke_out, all_segment_level_stroke_length, all_segment_level_term, all_segment_level_char, all_segment_level_char_length]
