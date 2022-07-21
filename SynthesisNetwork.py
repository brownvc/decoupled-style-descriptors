import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import math
import numpy as np
from helper import gaussian_2d
from config.GlobalVariables import *

class SynthesisNetwork(nn.Module):
	def __init__(self, weight_dim=512, num_layers=3, scale_sd=0, sentence_loss=True, word_loss=True, segment_loss=True, TYPE_A=True, TYPE_B=True, TYPE_C=True, TYPE_D=True, ORIGINAL=True, REC=True):
		super(SynthesisNetwork, self).__init__()
		self.num_mixtures				= 20
		self.num_layers					= num_layers
		self.weight_dim					= weight_dim
		self.device 					= 'cuda' if torch.cuda.is_available() else 'cpu'

		self.sentence_loss				= sentence_loss
		self.word_loss					= word_loss
		self.segment_loss				= segment_loss

		self.ORIGINAL 					= ORIGINAL
		self.TYPE_A						= TYPE_A
		self.TYPE_B						= TYPE_B
		self.TYPE_C 					= TYPE_C
		self.TYPE_D 					= TYPE_D
		self.REC						= REC

		self.magic_lstm					= nn.LSTM(self.weight_dim, self.weight_dim, batch_first=True, num_layers=self.num_layers)

		self.char_vec_fc_1				= nn.Linear(len(CHARACTERS), self.weight_dim)
		self.char_vec_relu_1 			= nn.LeakyReLU(negative_slope=0.1)
		self.char_lstm_1				= nn.LSTM(self.weight_dim, self.weight_dim, batch_first=True, num_layers=self.num_layers)
		self.char_vec_fc2_1 			= nn.Linear(self.weight_dim, self.weight_dim * self.weight_dim)

		# inference
		self.inf_state_fc1				= nn.Linear(3, self.weight_dim)
		self.inf_state_relu				= nn.LeakyReLU(negative_slope=0.1)
		self.inf_state_lstm				= nn.LSTM(self.weight_dim, self.weight_dim, batch_first=True, num_layers=self.num_layers)
		self.W_lstm						= nn.LSTM(self.weight_dim, self.weight_dim, batch_first=True, num_layers=self.num_layers)

		# generation
		self.gen_state_fc1				= nn.Linear(3, self.weight_dim)
		self.gen_state_relu				= nn.LeakyReLU(negative_slope=0.1)
		self.gen_state_lstm1			= nn.LSTM(self.weight_dim, self.weight_dim, batch_first=True, num_layers=self.num_layers)
		self.gen_state_lstm2			= nn.LSTM(self.weight_dim * 2, self.weight_dim * 2, batch_first=True, num_layers=self.num_layers)
		self.gen_state_fc2				= nn.Linear(self.weight_dim * 2, self.num_mixtures * 6 + 1)

		self.term_fc1					= nn.Linear(self.weight_dim * 2, self.weight_dim)
		self.term_relu1					= nn.LeakyReLU(negative_slope=0.1)
		self.term_fc2					= nn.Linear(self.weight_dim, self.weight_dim)
		self.term_relu2					= nn.LeakyReLU(negative_slope=0.1)
		self.term_fc3					= nn.Linear(self.weight_dim, 1)
		self.term_sigmoid				= nn.Sigmoid()

		self.mdn_sigmoid				= nn.Sigmoid()
		self.mdn_tanh					= nn.Tanh()
		self.mdn_softmax				= nn.Softmax(dim=1)
		self.scale_sd					= scale_sd

		self.mdn_bce_loss				= nn.BCEWithLogitsLoss()
		self.term_bce_loss				= nn.BCEWithLogitsLoss()

	def forward(self, inputs):
		[sentence_level_stroke_in, sentence_level_stroke_out, sentence_level_stroke_length, sentence_level_term, sentence_level_char, sentence_level_char_length, word_level_stroke_in, word_level_stroke_out, word_level_stroke_length, word_level_term, word_level_char, word_level_char_length, segment_level_stroke_in, segment_level_stroke_out, segment_level_stroke_length, segment_level_term, segment_level_char, segment_level_char_length] = inputs

		ALL_sentence_W_consistency_loss						= []

		ALL_ORIGINAL_sentence_termination_loss				= []
		ALL_ORIGINAL_sentence_loc_reconstruct_loss			= []
		ALL_ORIGINAL_sentence_touch_reconstruct_loss		= []

		ALL_TYPE_A_sentence_termination_loss				= []
		ALL_TYPE_A_sentence_loc_reconstruct_loss			= []
		ALL_TYPE_A_sentence_touch_reconstruct_loss			= []
		ALL_TYPE_A_sentence_WC_reconstruct_loss				= []

		ALL_TYPE_B_sentence_termination_loss				= []
		ALL_TYPE_B_sentence_loc_reconstruct_loss			= []
		ALL_TYPE_B_sentence_touch_reconstruct_loss			= []
		ALL_TYPE_B_sentence_WC_reconstruct_loss				= []


		ALL_word_W_consistency_loss							= []

		ALL_ORIGINAL_word_termination_loss					= []
		ALL_ORIGINAL_word_loc_reconstruct_loss				= []
		ALL_ORIGINAL_word_touch_reconstruct_loss			= []

		ALL_TYPE_A_word_termination_loss					= []
		ALL_TYPE_A_word_loc_reconstruct_loss				= []
		ALL_TYPE_A_word_touch_reconstruct_loss				= []
		ALL_TYPE_A_word_WC_reconstruct_loss					= []

		ALL_TYPE_B_word_termination_loss					= []
		ALL_TYPE_B_word_loc_reconstruct_loss				= []
		ALL_TYPE_B_word_touch_reconstruct_loss				= []
		ALL_TYPE_B_word_WC_reconstruct_loss					= []

		ALL_TYPE_C_word_termination_loss					= []
		ALL_TYPE_C_word_loc_reconstruct_loss				= []
		ALL_TYPE_C_word_touch_reconstruct_loss				= []
		ALL_TYPE_C_word_WC_reconstruct_loss					= []

		ALL_TYPE_D_word_termination_loss					= []
		ALL_TYPE_D_word_loc_reconstruct_loss				= []
		ALL_TYPE_D_word_touch_reconstruct_loss				= []
		ALL_TYPE_D_word_WC_reconstruct_loss					= []

		ALL_word_Wcs_reconstruct_TYPE_A						= []
		ALL_word_Wcs_reconstruct_TYPE_B						= []
		ALL_word_Wcs_reconstruct_TYPE_C						= []
		ALL_word_Wcs_reconstruct_TYPE_D						= []

		SUPER_ALL_segment_W_consistency_loss				= []

		SUPER_ALL_ORIGINAL_segment_termination_loss			= []
		SUPER_ALL_ORIGINAL_segment_loc_reconstruct_loss		= []
		SUPER_ALL_ORIGINAL_segment_touch_reconstruct_loss	= []

		SUPER_ALL_TYPE_A_segment_termination_loss			= []
		SUPER_ALL_TYPE_A_segment_loc_reconstruct_loss		= []
		SUPER_ALL_TYPE_A_segment_touch_reconstruct_loss		= []
		SUPER_ALL_TYPE_A_segment_WC_reconstruct_loss		= []

		SUPER_ALL_TYPE_B_segment_termination_loss			= []
		SUPER_ALL_TYPE_B_segment_loc_reconstruct_loss		= []
		SUPER_ALL_TYPE_B_segment_touch_reconstruct_loss		= []
		SUPER_ALL_TYPE_B_segment_WC_reconstruct_loss		= []

		SUPER_ALL_segment_Wcs_reconstruct_TYPE_A			= []
		SUPER_ALL_segment_Wcs_reconstruct_TYPE_B			= []

		# if self.sentece_loss:
		for uid in range(len(sentence_level_stroke_in)):
			if self.sentence_loss:
				user_sentence_level_stroke_in		= sentence_level_stroke_in[uid]
				user_sentence_level_stroke_out		= sentence_level_stroke_out[uid]
				user_sentence_level_stroke_length	= sentence_level_stroke_length[uid]
				user_sentence_level_term			= sentence_level_term[uid]
				user_sentence_level_char			= sentence_level_char[uid]
				user_sentence_level_char_length		= sentence_level_char_length[uid]

				sentence_batch_size					= len(user_sentence_level_stroke_in)

				sentence_inf_state_out				= self.inf_state_fc1(user_sentence_level_stroke_out)
				sentence_inf_state_out				= self.inf_state_relu(sentence_inf_state_out)
				sentence_inf_state_out, (c,h)		= self.inf_state_lstm(sentence_inf_state_out)

				sentence_gen_state_out				= self.gen_state_fc1(user_sentence_level_stroke_in)
				sentence_gen_state_out				= self.gen_state_relu(sentence_gen_state_out)
				sentence_gen_state_out, (c,h)		= self.gen_state_lstm1(sentence_gen_state_out)

				sentence_Ws							= []
				sentence_Wc_rec_TYPE_				= []
				sentence_SPLITS						= []
				sentence_Cs_1						= []
				sentence_unique_char_matrices_1		= []

				for sentence_batch_id in range(sentence_batch_size):
					curr_seq_len			= user_sentence_level_stroke_length[sentence_batch_id][0]
					curr_char_len			= user_sentence_level_char_length[sentence_batch_id][0]
					char_vector				= torch.eye(len(CHARACTERS))[user_sentence_level_char[sentence_batch_id][:curr_char_len]].to(self.device)
					current_term			= user_sentence_level_term[sentence_batch_id][:curr_seq_len].unsqueeze(-1)
					split_ids				= torch.nonzero(current_term)[:,0]

					char_vector_1				= self.char_vec_fc_1(char_vector)
					char_vector_1				= self.char_vec_relu_1(char_vector_1)

					unique_char_matrices_1			= []
					for cid in range(len(char_vector)):
						# Tower 1
						unique_char_vector_1		= char_vector_1[cid:cid+1]
						unique_char_input_1			= unique_char_vector_1.unsqueeze(0)
						unique_char_out_1, (c,h)	= self.char_lstm_1(unique_char_input_1)
						unique_char_out_1			= unique_char_out_1.squeeze(0)
						unique_char_out_1			= self.char_vec_fc2_1(unique_char_out_1)
						unique_char_matrix_1		= unique_char_out_1.view([-1,1,self.weight_dim,self.weight_dim])
						unique_char_matrix_1		= unique_char_matrix_1.squeeze(1)
						unique_char_matrices_1.append(unique_char_matrix_1)

					# Tower 1
					char_out_1				= char_vector_1.unsqueeze(0)
					char_out_1, (c,h) 		= self.char_lstm_1(char_out_1)
					char_out_1 				= char_out_1.squeeze(0)
					char_out_1				= self.char_vec_fc2_1(char_out_1)
					char_matrix_1			= char_out_1.view([-1,1,self.weight_dim,self.weight_dim])
					char_matrix_1			= char_matrix_1.squeeze(1)
					char_matrix_inv_1		= torch.inverse(char_matrix_1)

					W_c_t					= sentence_inf_state_out[sentence_batch_id][:curr_seq_len]
					W_c						= torch.stack([W_c_t[i] for i in split_ids])

					# W						= torch.bmm(char_matrix_inv, W_c.unsqueeze(2)).squeeze(-1)
					# C1C2C3W = Wc
					# W = C3-1 C2-1 C1-1 Wc
					W 						= torch.bmm(char_matrix_inv_1,
														  W_c.unsqueeze(2)).squeeze(-1)
					sentence_Ws.append(W)
					sentence_Wc_rec_TYPE_.append(W_c)
					sentence_Cs_1.append(char_matrix_1)
					sentence_SPLITS.append(split_ids)
					sentence_unique_char_matrices_1.append(unique_char_matrices_1)

				sentence_Ws_stacked				= torch.cat(sentence_Ws, 0)
				sentence_Ws_reshaped			= sentence_Ws_stacked.view([-1,self.weight_dim])
				sentence_W_mean					= sentence_Ws_reshaped.mean(0)
				sentence_W_mean_repeat			= sentence_W_mean.repeat(sentence_Ws_reshaped.size(0),1)
				sentence_Ws_consistency_loss	= torch.mean(torch.mean(torch.mul(sentence_W_mean_repeat - sentence_Ws_reshaped, sentence_W_mean_repeat - sentence_Ws_reshaped), -1))
				ALL_sentence_W_consistency_loss.append(sentence_Ws_consistency_loss)

				ORIGINAL_sentence_termination_loss			= []
				ORIGINAL_sentence_loc_reconstruct_loss		= []
				ORIGINAL_sentence_touch_reconstruct_loss	= []

				TYPE_A_sentence_termination_loss			= []
				TYPE_A_sentence_loc_reconstruct_loss		= []
				TYPE_A_sentence_touch_reconstruct_loss		= []

				TYPE_B_sentence_termination_loss			= []
				TYPE_B_sentence_loc_reconstruct_loss		= []
				TYPE_B_sentence_touch_reconstruct_loss		= []

				sentence_Wcs_reconstruct_TYPE_A				= []
				sentence_Wcs_reconstruct_TYPE_B				= []

				for sentence_batch_id in range(sentence_batch_size):

					sentence_level_gen_encoded		= sentence_gen_state_out[sentence_batch_id][:user_sentence_level_stroke_length[sentence_batch_id][0]]
					sentence_level_target_eos 		= user_sentence_level_stroke_out[sentence_batch_id][:user_sentence_level_stroke_length[sentence_batch_id][0]][:,2]
					sentence_level_target_x 		= user_sentence_level_stroke_out[sentence_batch_id][:user_sentence_level_stroke_length[sentence_batch_id][0]][:,0:1]
					sentence_level_target_y 		= user_sentence_level_stroke_out[sentence_batch_id][:user_sentence_level_stroke_length[sentence_batch_id][0]][:,1:2]
					sentence_level_target_term		= user_sentence_level_term[sentence_batch_id][:user_sentence_level_stroke_length[sentence_batch_id][0]]

					# ORIGINAL
					if self.ORIGINAL:
						sentence_W_lstm_in_ORIGINAL		= []
						curr_id							= 0
						for i in range(user_sentence_level_stroke_length[sentence_batch_id][0]):
							sentence_W_lstm_in_ORIGINAL.append(sentence_Wc_rec_TYPE_[sentence_batch_id][curr_id])
							if i in sentence_SPLITS[sentence_batch_id]:
								curr_id += 1
						sentence_W_lstm_in_ORIGINAL		= torch.stack(sentence_W_lstm_in_ORIGINAL)
						sentence_Wc_t_ORIGINAL			= sentence_W_lstm_in_ORIGINAL

						sentence_gen_lstm2_in_ORIGINAL	= torch.cat([sentence_level_gen_encoded, sentence_Wc_t_ORIGINAL], -1)
						sentence_gen_lstm2_in_ORIGINAL 	= sentence_gen_lstm2_in_ORIGINAL.unsqueeze(0)
						sentence_gen_out_ORIGINAL,(c,h) = self.gen_state_lstm2(sentence_gen_lstm2_in_ORIGINAL)
						sentence_gen_out_ORIGINAL		= sentence_gen_out_ORIGINAL.squeeze(0)

						mdn_out_ORIGINAL				= self.gen_state_fc2(sentence_gen_out_ORIGINAL)
						eos_ORIGINAL					= mdn_out_ORIGINAL[:,0:1]
						[mu1_ORIGINAL, mu2_ORIGINAL, sig1_ORIGINAL, sig2_ORIGINAL, rho_ORIGINAL, pi_ORIGINAL] = torch.split(mdn_out_ORIGINAL[:,1:], self.num_mixtures, 1)
						sig1_ORIGINAL					= sig1_ORIGINAL.exp() + 1e-3
						sig2_ORIGINAL					= sig2_ORIGINAL.exp() + 1e-3
						rho_ORIGINAL					= self.mdn_tanh(rho_ORIGINAL)
						pi_ORIGINAL						= self.mdn_softmax(pi_ORIGINAL)

						term_out_ORIGINAL				= self.term_fc1(sentence_gen_out_ORIGINAL)
						term_out_ORIGINAL				= self.term_relu1(term_out_ORIGINAL)
						term_out_ORIGINAL				= self.term_fc2(term_out_ORIGINAL)
						term_out_ORIGINAL				= self.term_relu2(term_out_ORIGINAL)
						term_out_ORIGINAL				= self.term_fc3(term_out_ORIGINAL)
						term_pred_ORIGINAL				= self.term_sigmoid(term_out_ORIGINAL)

						gaussian_ORIGINAL				= gaussian_2d(sentence_level_target_x, sentence_level_target_y, mu1_ORIGINAL, mu2_ORIGINAL, sig1_ORIGINAL, sig2_ORIGINAL, rho_ORIGINAL)
						loss_gaussian_ORIGINAL			= - torch.log(torch.sum(pi_ORIGINAL*gaussian_ORIGINAL, dim=1) + 1e-5)

						ORIGINAL_sentence_term_loss		= self.term_bce_loss(term_out_ORIGINAL.squeeze(1), sentence_level_target_term)
						ORIGINAL_sentence_loc_loss		= torch.mean(loss_gaussian_ORIGINAL)
						ORIGINAL_sentence_touch_loss	= self.mdn_bce_loss(eos_ORIGINAL.squeeze(1), sentence_level_target_eos)

						ORIGINAL_sentence_termination_loss.append(ORIGINAL_sentence_term_loss)
						ORIGINAL_sentence_loc_reconstruct_loss.append(ORIGINAL_sentence_loc_loss)
						ORIGINAL_sentence_touch_reconstruct_loss.append(ORIGINAL_sentence_touch_loss)

					# TYPE A
					if self.TYPE_A:
						sentence_C1 = sentence_Cs_1[sentence_batch_id]
						# sentence_Wc_rec_TYPE_A		= torch.bmm(sentence_Cs[sentence_batch_id], sentence_W_mean.repeat(sentence_Cs[sentence_batch_id].size(0),1).unsqueeze(2)).squeeze(-1)
						sentence_Wc_rec_TYPE_A		= 	torch.bmm(sentence_C1, \
																  sentence_W_mean.repeat(sentence_C1.size(0),1).unsqueeze(2)).squeeze(-1)

						sentence_Wcs_reconstruct_TYPE_A.append(sentence_Wc_rec_TYPE_A)

						sentence_W_lstm_in_TYPE_A		= []
						curr_id							= 0
						for i in range(user_sentence_level_stroke_length[sentence_batch_id][0]):
							sentence_W_lstm_in_TYPE_A.append(sentence_Wc_rec_TYPE_A[curr_id])
							if i in sentence_SPLITS[sentence_batch_id]:
								curr_id += 1
						sentence_Wc_t_rec_TYPE_A		= torch.stack(sentence_W_lstm_in_TYPE_A)

						sentence_gen_lstm2_in_TYPE_A	= torch.cat([sentence_level_gen_encoded, sentence_Wc_t_rec_TYPE_A], -1)
						sentence_gen_lstm2_in_TYPE_A 	= sentence_gen_lstm2_in_TYPE_A.unsqueeze(0)
						sentence_gen_out_TYPE_A, (c,h)	= self.gen_state_lstm2(sentence_gen_lstm2_in_TYPE_A)
						sentence_gen_out_TYPE_A			= sentence_gen_out_TYPE_A.squeeze(0)

						mdn_out_TYPE_A					= self.gen_state_fc2(sentence_gen_out_TYPE_A)
						eos_TYPE_A						= mdn_out_TYPE_A[:,0:1]
						[mu1_TYPE_A, mu2_TYPE_A, sig1_TYPE_A, sig2_TYPE_A, rho_TYPE_A, pi_TYPE_A] = torch.split(mdn_out_TYPE_A[:,1:], self.num_mixtures, 1)
						sig1_TYPE_A						= sig1_TYPE_A.exp() + 1e-3
						sig2_TYPE_A						= sig2_TYPE_A.exp() + 1e-3
						rho_TYPE_A						= self.mdn_tanh(rho_TYPE_A)
						pi_TYPE_A						= self.mdn_softmax(pi_TYPE_A)
						term_out_TYPE_A					= self.term_fc1(sentence_gen_out_TYPE_A)
						term_out_TYPE_A					= self.term_relu1(term_out_TYPE_A)
						term_out_TYPE_A					= self.term_fc2(term_out_TYPE_A)
						term_out_TYPE_A					= self.term_relu2(term_out_TYPE_A)
						term_out_TYPE_A					= self.term_fc3(term_out_TYPE_A)
						term_pred_TYPE_A				= self.term_sigmoid(term_out_TYPE_A)
						gaussian_TYPE_A					= gaussian_2d(sentence_level_target_x, sentence_level_target_y, mu1_TYPE_A, mu2_TYPE_A, sig1_TYPE_A, sig2_TYPE_A, rho_TYPE_A)
						loss_gaussian_TYPE_A			= - torch.log(torch.sum(pi_TYPE_A*gaussian_TYPE_A, dim=1) + 1e-5)

						TYPE_A_sentence_term_loss		= self.term_bce_loss(term_out_TYPE_A.squeeze(1), sentence_level_target_term)
						TYPE_A_sentence_loc_loss		= torch.mean(loss_gaussian_TYPE_A)
						TYPE_A_sentence_touch_loss		= self.mdn_bce_loss(eos_TYPE_A.squeeze(1), sentence_level_target_eos)

						TYPE_A_sentence_termination_loss.append(TYPE_A_sentence_term_loss)
						TYPE_A_sentence_loc_reconstruct_loss.append(TYPE_A_sentence_loc_loss)
						TYPE_A_sentence_touch_reconstruct_loss.append(TYPE_A_sentence_touch_loss)

					# TYPE B
					if self.TYPE_B:
						unique_char_matrix_1			= sentence_unique_char_matrices_1[sentence_batch_id]
						unique_char_matrices_1			= torch.stack(unique_char_matrix_1)
						unique_char_matrices_1			= unique_char_matrices_1.squeeze(1)

						# sentence_W_c_TYPE_B_RAW 		= torch.bmm(unique_char_matrices, sentence_W_mean.repeat(unique_char_matrices.size(0), 1).unsqueeze(2)).squeeze(-1)
						sentence_W_c_TYPE_B_RAW 		= torch.bmm(unique_char_matrices_1,
																sentence_W_mean.repeat(unique_char_matrices_1.size(0), 1).unsqueeze(2)).squeeze(-1)
						sentence_W_c_TYPE_B_RAW			= sentence_W_c_TYPE_B_RAW.unsqueeze(0)

						sentence_Wc_rec_TYPE_B, (c,h)	= self.magic_lstm(sentence_W_c_TYPE_B_RAW)
						sentence_Wc_rec_TYPE_B			= sentence_Wc_rec_TYPE_B.squeeze(0)

						sentence_Wcs_reconstruct_TYPE_B.append(sentence_Wc_rec_TYPE_B)

						sentence_W_lstm_in_TYPE_B		= []
						curr_id							= 0
						for i in range(user_sentence_level_stroke_length[sentence_batch_id][0]):
							sentence_W_lstm_in_TYPE_B.append(sentence_Wc_rec_TYPE_B[curr_id])
							if i in sentence_SPLITS[sentence_batch_id]:
								curr_id += 1
						sentence_Wc_t_rec_TYPE_B		= torch.stack(sentence_W_lstm_in_TYPE_B)

						sentence_gen_lstm2_in_TYPE_B	= torch.cat([sentence_level_gen_encoded, sentence_Wc_t_rec_TYPE_B], -1)
						sentence_gen_lstm2_in_TYPE_B 	= sentence_gen_lstm2_in_TYPE_B.unsqueeze(0)
						sentence_gen_out_TYPE_B, (c,h)	= self.gen_state_lstm2(sentence_gen_lstm2_in_TYPE_B)
						sentence_gen_out_TYPE_B			= sentence_gen_out_TYPE_B.squeeze(0)

						mdn_out_TYPE_B					= self.gen_state_fc2(sentence_gen_out_TYPE_B)
						eos_TYPE_B						= mdn_out_TYPE_B[:,0:1]
						[mu1_TYPE_B, mu2_TYPE_B, sig1_TYPE_B, sig2_TYPE_B, rho_TYPE_B, pi_TYPE_B] = torch.split(mdn_out_TYPE_B[:,1:], self.num_mixtures, 1)
						sig1_TYPE_B						= sig1_TYPE_B.exp() + 1e-3
						sig2_TYPE_B						= sig2_TYPE_B.exp() + 1e-3
						rho_TYPE_B						= self.mdn_tanh(rho_TYPE_B)
						pi_TYPE_B						= self.mdn_softmax(pi_TYPE_B)
						term_out_TYPE_B					= self.term_fc1(sentence_gen_out_TYPE_B)
						term_out_TYPE_B					= self.term_relu1(term_out_TYPE_B)
						term_out_TYPE_B					= self.term_fc2(term_out_TYPE_B)
						term_out_TYPE_B					= self.term_relu2(term_out_TYPE_B)
						term_out_TYPE_B					= self.term_fc3(term_out_TYPE_B)
						term_pred_TYPE_B				= self.term_sigmoid(term_out_TYPE_B)
						gaussian_TYPE_B					= gaussian_2d(sentence_level_target_x, sentence_level_target_y, mu1_TYPE_B, mu2_TYPE_B, sig1_TYPE_B, sig2_TYPE_B, rho_TYPE_B)
						loss_gaussian_TYPE_B			= - torch.log(torch.sum(pi_TYPE_B*gaussian_TYPE_B, dim=1) + 1e-5)

						TYPE_B_sentence_term_loss		= self.term_bce_loss(term_out_TYPE_B.squeeze(1), sentence_level_target_term)
						TYPE_B_sentence_loc_loss		= torch.mean(loss_gaussian_TYPE_B)
						TYPE_B_sentence_touch_loss		= self.mdn_bce_loss(eos_TYPE_B.squeeze(1), sentence_level_target_eos)

						TYPE_B_sentence_termination_loss.append(TYPE_B_sentence_term_loss)
						TYPE_B_sentence_loc_reconstruct_loss.append(TYPE_B_sentence_loc_loss)
						TYPE_B_sentence_touch_reconstruct_loss.append(TYPE_B_sentence_touch_loss)

				if self.ORIGINAL:
					ALL_ORIGINAL_sentence_termination_loss.append(torch.mean(torch.stack(ORIGINAL_sentence_termination_loss)))
					ALL_ORIGINAL_sentence_loc_reconstruct_loss.append(torch.mean(torch.stack(ORIGINAL_sentence_loc_reconstruct_loss)))
					ALL_ORIGINAL_sentence_touch_reconstruct_loss.append(torch.mean(torch.stack(ORIGINAL_sentence_touch_reconstruct_loss)))

				if self.TYPE_A:
					ALL_TYPE_A_sentence_termination_loss.append(torch.mean(torch.stack(TYPE_A_sentence_termination_loss)))
					ALL_TYPE_A_sentence_loc_reconstruct_loss.append(torch.mean(torch.stack(TYPE_A_sentence_loc_reconstruct_loss)))
					ALL_TYPE_A_sentence_touch_reconstruct_loss.append(torch.mean(torch.stack(TYPE_A_sentence_touch_reconstruct_loss)))

					if self.REC:
						TYPE_A_sentence_WC_reconstruct_loss	= []
						for sentence_batch_id in range(len(sentence_Wc_rec_TYPE_)):
							sentence_Wc_ORIGINAL	= sentence_Wc_rec_TYPE_[sentence_batch_id]
							sentence_Wc_TYPE_A		= sentence_Wcs_reconstruct_TYPE_A[sentence_batch_id]
							sentence_WC_reconstruct_loss_TYPE_A	= torch.mean(torch.mean(torch.mul(sentence_Wc_ORIGINAL - sentence_Wc_TYPE_A, sentence_Wc_ORIGINAL - sentence_Wc_TYPE_A), -1))
							TYPE_A_sentence_WC_reconstruct_loss.append(sentence_WC_reconstruct_loss_TYPE_A)
						ALL_TYPE_A_sentence_WC_reconstruct_loss.append(torch.mean(torch.stack(TYPE_A_sentence_WC_reconstruct_loss)))

				if self.TYPE_B:
					ALL_TYPE_B_sentence_termination_loss.append(torch.mean(torch.stack(TYPE_B_sentence_termination_loss)))
					ALL_TYPE_B_sentence_loc_reconstruct_loss.append(torch.mean(torch.stack(TYPE_B_sentence_loc_reconstruct_loss)))
					ALL_TYPE_B_sentence_touch_reconstruct_loss.append(torch.mean(torch.stack(TYPE_B_sentence_touch_reconstruct_loss)))

					if self.REC:
						TYPE_B_sentence_WC_reconstruct_loss	= []
						for sentence_batch_id in range(len(sentence_Wc_rec_TYPE_)):
							sentence_Wc_ORIGINAL	= sentence_Wc_rec_TYPE_[sentence_batch_id]
							sentence_Wc_TYPE_B		= sentence_Wcs_reconstruct_TYPE_B[sentence_batch_id]
							sentence_WC_reconstruct_loss_TYPE_B	= torch.mean(torch.mean(torch.mul(sentence_Wc_ORIGINAL - sentence_Wc_TYPE_B, sentence_Wc_ORIGINAL - sentence_Wc_TYPE_B), -1))
							TYPE_B_sentence_WC_reconstruct_loss.append(sentence_WC_reconstruct_loss_TYPE_B)
						ALL_TYPE_B_sentence_WC_reconstruct_loss.append(torch.mean(torch.stack(TYPE_B_sentence_WC_reconstruct_loss)))

			if self.word_loss:
				user_word_level_stroke_in		= word_level_stroke_in[uid]
				user_word_level_stroke_out		= word_level_stroke_out[uid]
				user_word_level_stroke_length	= word_level_stroke_length[uid]
				user_word_level_term			= word_level_term[uid]
				user_word_level_char			= word_level_char[uid]
				user_word_level_char_length		= word_level_char_length[uid]

				word_batch_size					= len(user_word_level_stroke_in)

				word_inf_state_out				= self.inf_state_fc1(user_word_level_stroke_out)
				word_inf_state_out				= self.inf_state_relu(word_inf_state_out)
				word_inf_state_out, (c,h)		= self.inf_state_lstm(word_inf_state_out)

				word_gen_state_out				= self.gen_state_fc1(user_word_level_stroke_in)
				word_gen_state_out				= self.gen_state_relu(word_gen_state_out)
				word_gen_state_out, (c,h)		= self.gen_state_lstm1(word_gen_state_out)

				word_Ws							= []
				word_Wc_rec_ORIGINAL			= []
				word_SPLITS						= []
				word_Cs_1						= []
				word_unique_char_matrices_1		= []

				W_C_ORIGINALS	= []
				for word_batch_id in range(word_batch_size):
					curr_seq_len			= user_word_level_stroke_length[word_batch_id][0]
					curr_char_len			= user_word_level_char_length[word_batch_id][0]
					char_vector				= torch.eye(len(CHARACTERS))[user_word_level_char[word_batch_id][:curr_char_len]].to(self.device)
					current_term			= user_word_level_term[word_batch_id][:curr_seq_len].unsqueeze(-1)
					split_ids				= torch.nonzero(current_term)[:,0]

					char_vector_1				= self.char_vec_fc_1(char_vector)
					char_vector_1				= self.char_vec_relu_1(char_vector_1)

					unique_char_matrices_1		= []
					for cid in range(len(char_vector)):
						# Tower 1
						unique_char_vector_1		= char_vector_1[cid:cid+1]
						unique_char_input_1			= unique_char_vector_1.unsqueeze(0)
						unique_char_out_1, (c,h)	= self.char_lstm_1(unique_char_input_1)
						unique_char_out_1			= unique_char_out_1.squeeze(0)
						unique_char_out_1			= self.char_vec_fc2_1(unique_char_out_1)
						unique_char_matrix_1		= unique_char_out_1.view([-1,1,self.weight_dim,self.weight_dim])
						unique_char_matrix_1		= unique_char_matrix_1.squeeze(1)
						unique_char_matrices_1.append(unique_char_matrix_1)

					# Tower 1
					char_out_1				= char_vector_1.unsqueeze(0)
					char_out_1, (c,h) 		= self.char_lstm_1(char_out_1)
					char_out_1 				= char_out_1.squeeze(0)
					char_out_1				= self.char_vec_fc2_1(char_out_1)
					char_matrix_1			= char_out_1.view([-1,1,self.weight_dim,self.weight_dim])
					char_matrix_1			= char_matrix_1.squeeze(1)
					char_matrix_inv_1		= torch.inverse(char_matrix_1)

					W_c_t					= word_inf_state_out[word_batch_id][:curr_seq_len]
					W_c						= torch.stack([W_c_t[i] for i in split_ids])

					W_C_ORIGINAL	= {}
					for i in range(curr_char_len):
						sub_s = "".join(CHARACTERS[i] for i in user_word_level_char[word_batch_id][:i+1])
						W_C_ORIGINAL[sub_s] = [W_c[i]]
					W_C_ORIGINALS.append(W_C_ORIGINAL)

					# W						= torch.bmm(char_matrix_inv, W_c.unsqueeze(2)).squeeze(-1)
					W 						= torch.bmm(char_matrix_inv_1,
														  W_c.unsqueeze(2)).squeeze(-1)
					word_Ws.append(W)
					word_Wc_rec_ORIGINAL.append(W_c)
					word_SPLITS.append(split_ids)
					# word_Cs.append(char_matrix)
					# word_unique_char_matrices.append(unique_char_matrices)
					word_Cs_1.append(char_matrix_1)
					word_unique_char_matrices_1.append(unique_char_matrices_1)

				word_Ws_stacked					= torch.cat(word_Ws, 0)
				word_Ws_reshaped				= word_Ws_stacked.view([-1,self.weight_dim])
				word_W_mean						= word_Ws_reshaped.mean(0)
				word_Ws_reshaped_mean_repeat	= word_W_mean.repeat(word_Ws_reshaped.size(0),1)
				word_Ws_consistency_loss		= torch.mean(torch.mean(torch.mul(word_Ws_reshaped_mean_repeat - word_Ws_reshaped, word_Ws_reshaped_mean_repeat - word_Ws_reshaped), -1))
				ALL_word_W_consistency_loss.append(word_Ws_consistency_loss)

				# word
				ORIGINAL_word_termination_loss				= []
				ORIGINAL_word_loc_reconstruct_loss			= []
				ORIGINAL_word_touch_reconstruct_loss		= []

				TYPE_A_word_termination_loss				= []
				TYPE_A_word_loc_reconstruct_loss			= []
				TYPE_A_word_touch_reconstruct_loss			= []

				TYPE_B_word_termination_loss				= []
				TYPE_B_word_loc_reconstruct_loss			= []
				TYPE_B_word_touch_reconstruct_loss			= []

				TYPE_C_word_termination_loss				= []
				TYPE_C_word_loc_reconstruct_loss			= []
				TYPE_C_word_touch_reconstruct_loss			= []

				TYPE_D_word_termination_loss				= []
				TYPE_D_word_loc_reconstruct_loss			= []
				TYPE_D_word_touch_reconstruct_loss			= []

				word_Wcs_reconstruct_TYPE_A					= []
				word_Wcs_reconstruct_TYPE_B					= []
				word_Wcs_reconstruct_TYPE_C					= []
				word_Wcs_reconstruct_TYPE_D					= []

				# segment

				ALL_segment_W_consistency_loss				= []

				ALL_ORIGINAL_segment_termination_loss		= []
				ALL_ORIGINAL_segment_loc_reconstruct_loss	= []
				ALL_ORIGINAL_segment_touch_reconstruct_loss	= []

				ALL_TYPE_A_segment_termination_loss			= []
				ALL_TYPE_A_segment_loc_reconstruct_loss		= []
				ALL_TYPE_A_segment_touch_reconstruct_loss	= []
				ALL_TYPE_A_segment_WC_reconstruct_loss		= []

				ALL_TYPE_B_segment_termination_loss			= []
				ALL_TYPE_B_segment_loc_reconstruct_loss		= []
				ALL_TYPE_B_segment_touch_reconstruct_loss	= []
				ALL_TYPE_B_segment_WC_reconstruct_loss		= []

				ALL_segment_Wcs_reconstruct_TYPE_A			= []
				ALL_segment_Wcs_reconstruct_TYPE_B			= []

				W_C_SEGMENTS	= []
				W_C_UNIQUES		= []
				for word_batch_id in range(word_batch_size):

					word_level_gen_encoded		= word_gen_state_out[word_batch_id][:user_word_level_stroke_length[word_batch_id][0]]
					word_level_target_eos 		= user_word_level_stroke_out[word_batch_id][:user_word_level_stroke_length[word_batch_id][0]][:,2]
					word_level_target_x 		= user_word_level_stroke_out[word_batch_id][:user_word_level_stroke_length[word_batch_id][0]][:,0:1]
					word_level_target_y 		= user_word_level_stroke_out[word_batch_id][:user_word_level_stroke_length[word_batch_id][0]][:,1:2]
					word_level_target_term		= user_word_level_term[word_batch_id][:user_word_level_stroke_length[word_batch_id][0]]

					# ORIGINAL
					if self.ORIGINAL:
						word_W_lstm_in_ORIGINAL		= []
						curr_id							= 0
						for i in range(user_word_level_stroke_length[word_batch_id][0]):
							word_W_lstm_in_ORIGINAL.append(word_Wc_rec_ORIGINAL[word_batch_id][curr_id])
							if i in word_SPLITS[word_batch_id]:
								curr_id += 1
						word_W_lstm_in_ORIGINAL		= torch.stack(word_W_lstm_in_ORIGINAL)
						word_Wc_t_ORIGINAL			= word_W_lstm_in_ORIGINAL

						word_gen_lstm2_in_ORIGINAL	= torch.cat([word_level_gen_encoded, word_Wc_t_ORIGINAL], -1)
						word_gen_lstm2_in_ORIGINAL 	= word_gen_lstm2_in_ORIGINAL.unsqueeze(0)
						word_gen_out_ORIGINAL,(c,h) = self.gen_state_lstm2(word_gen_lstm2_in_ORIGINAL)
						word_gen_out_ORIGINAL		= word_gen_out_ORIGINAL.squeeze(0)

						mdn_out_ORIGINAL				= self.gen_state_fc2(word_gen_out_ORIGINAL)
						eos_ORIGINAL					= mdn_out_ORIGINAL[:,0:1]
						[mu1_ORIGINAL, mu2_ORIGINAL, sig1_ORIGINAL, sig2_ORIGINAL, rho_ORIGINAL, pi_ORIGINAL] = torch.split(mdn_out_ORIGINAL[:,1:], self.num_mixtures, 1)
						sig1_ORIGINAL					= sig1_ORIGINAL.exp() + 1e-3
						sig2_ORIGINAL					= sig2_ORIGINAL.exp() + 1e-3
						rho_ORIGINAL					= self.mdn_tanh(rho_ORIGINAL)
						pi_ORIGINAL						= self.mdn_softmax(pi_ORIGINAL)

						term_out_ORIGINAL				= self.term_fc1(word_gen_out_ORIGINAL)
						term_out_ORIGINAL				= self.term_relu1(term_out_ORIGINAL)
						term_out_ORIGINAL				= self.term_fc2(term_out_ORIGINAL)
						term_out_ORIGINAL				= self.term_relu2(term_out_ORIGINAL)
						term_out_ORIGINAL				= self.term_fc3(term_out_ORIGINAL)
						term_pred_ORIGINAL				= self.term_sigmoid(term_out_ORIGINAL)

						gaussian_ORIGINAL				= gaussian_2d(word_level_target_x, word_level_target_y, mu1_ORIGINAL, mu2_ORIGINAL, sig1_ORIGINAL, sig2_ORIGINAL, rho_ORIGINAL)
						loss_gaussian_ORIGINAL			= - torch.log(torch.sum(pi_ORIGINAL*gaussian_ORIGINAL, dim=1) + 1e-5)

						ORIGINAL_word_term_loss		= self.term_bce_loss(term_out_ORIGINAL.squeeze(1), word_level_target_term)
						ORIGINAL_word_loc_loss		= torch.mean(loss_gaussian_ORIGINAL)
						ORIGINAL_word_touch_loss	= self.mdn_bce_loss(eos_ORIGINAL.squeeze(1), word_level_target_eos)

						ORIGINAL_word_termination_loss.append(ORIGINAL_word_term_loss)
						ORIGINAL_word_loc_reconstruct_loss.append(ORIGINAL_word_loc_loss)
						ORIGINAL_word_touch_reconstruct_loss.append(ORIGINAL_word_touch_loss)

					# TYPE A
					if self.TYPE_A:
						word_C1 = word_Cs_1[word_batch_id]
						word_Wc_rec_TYPE_A		= 	torch.bmm(word_C1,
															  word_W_mean.repeat(word_C1.size(0),1).unsqueeze(2)).squeeze(-1)

						word_Wcs_reconstruct_TYPE_A.append(word_Wc_rec_TYPE_A)

						word_W_lstm_in_TYPE_A		= []
						curr_id					= 0
						for i in range(user_word_level_stroke_length[word_batch_id][0]):
							word_W_lstm_in_TYPE_A.append(word_Wc_rec_TYPE_A[curr_id])
							if i in word_SPLITS[word_batch_id]:
								curr_id += 1
						word_Wc_t_rec_TYPE_A		= torch.stack(word_W_lstm_in_TYPE_A)

						word_gen_lstm2_in_TYPE_A	= torch.cat([word_level_gen_encoded, word_Wc_t_rec_TYPE_A], -1)
						word_gen_lstm2_in_TYPE_A 	= word_gen_lstm2_in_TYPE_A.unsqueeze(0)
						word_gen_out_TYPE_A, (c,h)	= self.gen_state_lstm2(word_gen_lstm2_in_TYPE_A)
						word_gen_out_TYPE_A			= word_gen_out_TYPE_A.squeeze(0)

						mdn_out_TYPE_A					= self.gen_state_fc2(word_gen_out_TYPE_A)
						eos_TYPE_A						= mdn_out_TYPE_A[:,0:1]
						[mu1_TYPE_A, mu2_TYPE_A, sig1_TYPE_A, sig2_TYPE_A, rho_TYPE_A, pi_TYPE_A] = torch.split(mdn_out_TYPE_A[:,1:], self.num_mixtures, 1)
						sig1_TYPE_A						= sig1_TYPE_A.exp() + 1e-3
						sig2_TYPE_A						= sig2_TYPE_A.exp() + 1e-3
						rho_TYPE_A						= self.mdn_tanh(rho_TYPE_A)
						pi_TYPE_A						= self.mdn_softmax(pi_TYPE_A)
						term_out_TYPE_A					= self.term_fc1(word_gen_out_TYPE_A)
						term_out_TYPE_A					= self.term_relu1(term_out_TYPE_A)
						term_out_TYPE_A					= self.term_fc2(term_out_TYPE_A)
						term_out_TYPE_A					= self.term_relu2(term_out_TYPE_A)
						term_out_TYPE_A					= self.term_fc3(term_out_TYPE_A)
						term_pred_TYPE_A				= self.term_sigmoid(term_out_TYPE_A)
						gaussian_TYPE_A					= gaussian_2d(word_level_target_x, word_level_target_y, mu1_TYPE_A, mu2_TYPE_A, sig1_TYPE_A, sig2_TYPE_A, rho_TYPE_A)
						loss_gaussian_TYPE_A			= - torch.log(torch.sum(pi_TYPE_A*gaussian_TYPE_A, dim=1) + 1e-5)

						TYPE_A_word_term_loss		= self.term_bce_loss(term_out_TYPE_A.squeeze(1), word_level_target_term)
						TYPE_A_word_loc_loss		= torch.mean(loss_gaussian_TYPE_A)
						TYPE_A_word_touch_loss		= self.mdn_bce_loss(eos_TYPE_A.squeeze(1), word_level_target_eos)

						TYPE_A_word_termination_loss.append(TYPE_A_word_term_loss)
						TYPE_A_word_loc_reconstruct_loss.append(TYPE_A_word_loc_loss)
						TYPE_A_word_touch_reconstruct_loss.append(TYPE_A_word_touch_loss)

					# TYPE B
					if self.TYPE_B:
						unique_char_matrix_1		= word_unique_char_matrices_1[word_batch_id]
						unique_char_matrices_1		= torch.stack(unique_char_matrix_1)
						unique_char_matrices_1		= unique_char_matrices_1.squeeze(1)

						# word_W_c_TYPE_B_RAW 		= torch.bmm(unique_char_matrices, word_W_mean.repeat(unique_char_matrices.size(0), 1).unsqueeze(2)).squeeze(-1)
						word_W_c_TYPE_B_RAW 		= torch.bmm(unique_char_matrices_1,
																word_W_mean.repeat(unique_char_matrices_1.size(0), 1).unsqueeze(2)).squeeze(-1)
						word_W_c_TYPE_B_RAW			= word_W_c_TYPE_B_RAW.unsqueeze(0)

						word_Wc_rec_TYPE_B, (c,h)	= self.magic_lstm(word_W_c_TYPE_B_RAW)
						word_Wc_rec_TYPE_B			= word_Wc_rec_TYPE_B.squeeze(0)

						word_Wcs_reconstruct_TYPE_B.append(word_Wc_rec_TYPE_B)

						word_W_lstm_in_TYPE_B		= []
						curr_id							= 0
						for i in range(user_word_level_stroke_length[word_batch_id][0]):
							word_W_lstm_in_TYPE_B.append(word_Wc_rec_TYPE_B[curr_id])
							if i in word_SPLITS[word_batch_id]:
								curr_id += 1
						word_Wc_t_rec_TYPE_B		= torch.stack(word_W_lstm_in_TYPE_B)
						word_gen_lstm2_in_TYPE_B	= torch.cat([word_level_gen_encoded, word_Wc_t_rec_TYPE_B], -1)
						word_gen_lstm2_in_TYPE_B 	= word_gen_lstm2_in_TYPE_B.unsqueeze(0)
						word_gen_out_TYPE_B, (c,h)	= self.gen_state_lstm2(word_gen_lstm2_in_TYPE_B)
						word_gen_out_TYPE_B			= word_gen_out_TYPE_B.squeeze(0)

						mdn_out_TYPE_B					= self.gen_state_fc2(word_gen_out_TYPE_B)
						eos_TYPE_B						= mdn_out_TYPE_B[:,0:1]
						[mu1_TYPE_B, mu2_TYPE_B, sig1_TYPE_B, sig2_TYPE_B, rho_TYPE_B, pi_TYPE_B] = torch.split(mdn_out_TYPE_B[:,1:], self.num_mixtures, 1)
						sig1_TYPE_B						= sig1_TYPE_B.exp() + 1e-3
						sig2_TYPE_B						= sig2_TYPE_B.exp() + 1e-3
						rho_TYPE_B						= self.mdn_tanh(rho_TYPE_B)
						pi_TYPE_B						= self.mdn_softmax(pi_TYPE_B)
						term_out_TYPE_B					= self.term_fc1(word_gen_out_TYPE_B)
						term_out_TYPE_B					= self.term_relu1(term_out_TYPE_B)
						term_out_TYPE_B					= self.term_fc2(term_out_TYPE_B)
						term_out_TYPE_B					= self.term_relu2(term_out_TYPE_B)
						term_out_TYPE_B					= self.term_fc3(term_out_TYPE_B)
						term_pred_TYPE_B				= self.term_sigmoid(term_out_TYPE_B)
						gaussian_TYPE_B					= gaussian_2d(word_level_target_x, word_level_target_y, mu1_TYPE_B, mu2_TYPE_B, sig1_TYPE_B, sig2_TYPE_B, rho_TYPE_B)
						loss_gaussian_TYPE_B			= - torch.log(torch.sum(pi_TYPE_B*gaussian_TYPE_B, dim=1) + 1e-5)

						TYPE_B_word_term_loss		= self.term_bce_loss(term_out_TYPE_B.squeeze(1), word_level_target_term)
						TYPE_B_word_loc_loss		= torch.mean(loss_gaussian_TYPE_B)
						TYPE_B_word_touch_loss		= self.mdn_bce_loss(eos_TYPE_B.squeeze(1), word_level_target_eos)

						TYPE_B_word_termination_loss.append(TYPE_B_word_term_loss)
						TYPE_B_word_loc_reconstruct_loss.append(TYPE_B_word_loc_loss)
						TYPE_B_word_touch_reconstruct_loss.append(TYPE_B_word_touch_loss)

					# TYPE C
					# if self.TYPE_C:
					user_segment_level_stroke_in		= segment_level_stroke_in[uid][word_batch_id]
					user_segment_level_stroke_out		= segment_level_stroke_out[uid][word_batch_id]
					user_segment_level_stroke_length	= segment_level_stroke_length[uid][word_batch_id]
					user_segment_level_term				= segment_level_term[uid][word_batch_id]
					user_segment_level_char				= segment_level_char[uid][word_batch_id]
					user_segment_level_char_length		= segment_level_char_length[uid][word_batch_id]

					segment_batch_size					= len(user_segment_level_stroke_in)

					segment_inf_state_out				= self.inf_state_fc1(user_segment_level_stroke_out)
					segment_inf_state_out				= self.inf_state_relu(segment_inf_state_out)
					segment_inf_state_out, (c,h)		= self.inf_state_lstm(segment_inf_state_out)

					segment_gen_state_out				= self.gen_state_fc1(user_segment_level_stroke_in)
					segment_gen_state_out				= self.gen_state_relu(segment_gen_state_out)
					segment_gen_state_out, (c,h)		= self.gen_state_lstm1(segment_gen_state_out)

					segment_Ws							= []
					segment_Wc_rec_ORIGINAL				= []
					segment_SPLITS						= []
					segment_Cs_1						= []
					segment_unique_char_matrices_1		= []

					W_C_SEGMENT = {}

					for segment_batch_id in range(segment_batch_size):
						curr_seq_len			= user_segment_level_stroke_length[segment_batch_id][0]
						curr_char_len			= user_segment_level_char_length[segment_batch_id][0]
						char_vector				= torch.eye(len(CHARACTERS))[user_segment_level_char[segment_batch_id][:curr_char_len]].to(self.device)
						current_term			= user_segment_level_term[segment_batch_id][:curr_seq_len].unsqueeze(-1)
						split_ids				= torch.nonzero(current_term)[:,0]

						char_vector_1			= self.char_vec_fc_1(char_vector)
						char_vector_1			= self.char_vec_relu_1(char_vector_1)
						unique_char_matrices_1	= []

						for cid in range(len(char_vector)):
							# Tower 1
							unique_char_vector_1		= char_vector_1[cid:cid+1]
							unique_char_input_1			= unique_char_vector_1.unsqueeze(0)
							unique_char_out_1, (c,h)	= self.char_lstm_1(unique_char_input_1)
							unique_char_out_1			= unique_char_out_1.squeeze(0)
							unique_char_out_1			= self.char_vec_fc2_1(unique_char_out_1)
							unique_char_matrix_1		= unique_char_out_1.view([-1,1,self.weight_dim,self.weight_dim])
							unique_char_matrix_1		= unique_char_matrix_1.squeeze(1)
							unique_char_matrices_1.append(unique_char_matrix_1)

						# Tower 1
						char_out_1				= char_vector_1.unsqueeze(0)
						char_out_1, (c,h) 		= self.char_lstm_1(char_out_1)
						char_out_1 				= char_out_1.squeeze(0)
						char_out_1				= self.char_vec_fc2_1(char_out_1)
						char_matrix_1			= char_out_1.view([-1,1,self.weight_dim,self.weight_dim])
						char_matrix_1			= char_matrix_1.squeeze(1)
						char_matrix_inv_1		= torch.inverse(char_matrix_1)

						W_c_t					= segment_inf_state_out[segment_batch_id][:curr_seq_len]
						W_c						= torch.stack([W_c_t[i] for i in split_ids])

						for i in range(curr_char_len):
							sub_s = "".join(CHARACTERS[i] for i in user_segment_level_char[segment_batch_id][:i+1])
							if sub_s in W_C_SEGMENT:
								W_C_SEGMENT[sub_s].append(W_c[i])
							else:
								W_C_SEGMENT[sub_s] = [W_c[i]]

						W 						= torch.bmm(char_matrix_inv_1,
															  W_c.unsqueeze(2)).squeeze(-1)
						segment_Ws.append(W)
						segment_Wc_rec_ORIGINAL.append(W_c)
						segment_SPLITS.append(split_ids)
						segment_Cs_1.append(char_matrix_1)
						segment_unique_char_matrices_1.append(unique_char_matrices_1)

					W_C_SEGMENTS.append(W_C_SEGMENT)

					if self.segment_loss:
						segment_Ws_stacked				= torch.cat(segment_Ws, 0)
						segment_Ws_reshaped				= segment_Ws_stacked.view([-1,self.weight_dim])
						segment_W_mean					= segment_Ws_reshaped.mean(0)
						segment_Ws_reshaped_mean_repeat	= segment_W_mean.repeat(segment_Ws_reshaped.size(0),1)
						segment_Ws_consistency_loss		= torch.mean(torch.mean(torch.mul(segment_Ws_reshaped_mean_repeat - segment_Ws_reshaped, segment_Ws_reshaped_mean_repeat - segment_Ws_reshaped), -1))
						ALL_segment_W_consistency_loss.append(segment_Ws_consistency_loss)

						ORIGINAL_segment_termination_loss		= []
						ORIGINAL_segment_loc_reconstruct_loss	= []
						ORIGINAL_segment_touch_reconstruct_loss	= []

						TYPE_A_segment_termination_loss			= []
						TYPE_A_segment_loc_reconstruct_loss		= []
						TYPE_A_segment_touch_reconstruct_loss	= []

						TYPE_B_segment_termination_loss			= []
						TYPE_B_segment_loc_reconstruct_loss		= []
						TYPE_B_segment_touch_reconstruct_loss	= []

						segment_Wcs_reconstruct_TYPE_A			= []
						segment_Wcs_reconstruct_TYPE_B			= []

						for segment_batch_id in range(segment_batch_size):
							segment_level_gen_encoded			= segment_gen_state_out[segment_batch_id][:user_segment_level_stroke_length[segment_batch_id][0]]
							segment_level_target_eos 			= user_segment_level_stroke_out[segment_batch_id][:user_segment_level_stroke_length[segment_batch_id][0]][:,2]
							segment_level_target_x 				= user_segment_level_stroke_out[segment_batch_id][:user_segment_level_stroke_length[segment_batch_id][0]][:,0:1]
							segment_level_target_y 				= user_segment_level_stroke_out[segment_batch_id][:user_segment_level_stroke_length[segment_batch_id][0]][:,1:2]
							segment_level_target_term			= user_segment_level_term[segment_batch_id][:user_segment_level_stroke_length[segment_batch_id][0]]

							if self.ORIGINAL:
								segment_W_lstm_in_ORIGINAL		= []
								curr_id						= 0
								for i in range(user_segment_level_stroke_length[segment_batch_id][0]):
									segment_W_lstm_in_ORIGINAL.append(segment_Wc_rec_ORIGINAL[segment_batch_id][curr_id])
									if i in segment_SPLITS[segment_batch_id]:
										curr_id += 1
								segment_W_lstm_in_ORIGINAL		= torch.stack(segment_W_lstm_in_ORIGINAL)
								segment_Wc_t_ORIGINAL			= segment_W_lstm_in_ORIGINAL

								segment_gen_lstm2_in_ORIGINAL	= torch.cat([segment_level_gen_encoded, segment_Wc_t_ORIGINAL], -1)
								segment_gen_lstm2_in_ORIGINAL 	= segment_gen_lstm2_in_ORIGINAL.unsqueeze(0)
								segment_gen_out_ORIGINAL,(c,h) = self.gen_state_lstm2(segment_gen_lstm2_in_ORIGINAL)
								segment_gen_out_ORIGINAL		= segment_gen_out_ORIGINAL.squeeze(0)

								mdn_out_ORIGINAL				= self.gen_state_fc2(segment_gen_out_ORIGINAL)
								eos_ORIGINAL					= mdn_out_ORIGINAL[:,0:1]
								[mu1_ORIGINAL, mu2_ORIGINAL, sig1_ORIGINAL, sig2_ORIGINAL, rho_ORIGINAL, pi_ORIGINAL] = torch.split(mdn_out_ORIGINAL[:,1:], self.num_mixtures, 1)
								sig1_ORIGINAL					= sig1_ORIGINAL.exp() + 1e-3
								sig2_ORIGINAL					= sig2_ORIGINAL.exp() + 1e-3
								rho_ORIGINAL					= self.mdn_tanh(rho_ORIGINAL)
								pi_ORIGINAL					= self.mdn_softmax(pi_ORIGINAL)

								term_out_ORIGINAL				= self.term_fc1(segment_gen_out_ORIGINAL)
								term_out_ORIGINAL				= self.term_relu1(term_out_ORIGINAL)
								term_out_ORIGINAL				= self.term_fc2(term_out_ORIGINAL)
								term_out_ORIGINAL				= self.term_relu2(term_out_ORIGINAL)
								term_out_ORIGINAL				= self.term_fc3(term_out_ORIGINAL)
								term_pred_ORIGINAL				= self.term_sigmoid(term_out_ORIGINAL)

								gaussian_ORIGINAL				= gaussian_2d(segment_level_target_x, segment_level_target_y, mu1_ORIGINAL, mu2_ORIGINAL, sig1_ORIGINAL, sig2_ORIGINAL, rho_ORIGINAL)
								loss_gaussian_ORIGINAL			= - torch.log(torch.sum(pi_ORIGINAL*gaussian_ORIGINAL, dim=1) + 1e-5)

								ORIGINAL_segment_term_loss		= self.term_bce_loss(term_out_ORIGINAL.squeeze(1), segment_level_target_term)
								ORIGINAL_segment_loc_loss		= torch.mean(loss_gaussian_ORIGINAL)
								ORIGINAL_segment_touch_loss	= self.mdn_bce_loss(eos_ORIGINAL.squeeze(1), segment_level_target_eos)

								ORIGINAL_segment_termination_loss.append(ORIGINAL_segment_term_loss)
								ORIGINAL_segment_loc_reconstruct_loss.append(ORIGINAL_segment_loc_loss)
								ORIGINAL_segment_touch_reconstruct_loss.append(ORIGINAL_segment_touch_loss)

							# TYPE A
							if self.TYPE_A:
								segment_C1 = segment_Cs_1[segment_batch_id]
								segment_Wc_rec_TYPE_A			= torch.bmm(segment_C1,
																segment_W_mean.repeat(segment_C1.size(0),1).unsqueeze(2)).squeeze(-1)
								segment_Wcs_reconstruct_TYPE_A.append(segment_Wc_rec_TYPE_A)

								segment_W_lstm_in_TYPE_A		= []
								curr_id							= 0
								for i in range(user_segment_level_stroke_length[segment_batch_id][0]):
									segment_W_lstm_in_TYPE_A.append(segment_Wc_rec_TYPE_A[curr_id])
									if i in segment_SPLITS[segment_batch_id]:
										curr_id += 1
								segment_Wc_t_rec_TYPE_A			= torch.stack(segment_W_lstm_in_TYPE_A)

								segment_gen_lstm2_in_TYPE_A		= torch.cat([segment_level_gen_encoded, segment_Wc_t_rec_TYPE_A], -1)
								segment_gen_lstm2_in_TYPE_A 	= segment_gen_lstm2_in_TYPE_A.unsqueeze(0)
								segment_gen_out_TYPE_A, (c,h)	= self.gen_state_lstm2(segment_gen_lstm2_in_TYPE_A)
								segment_gen_out_TYPE_A			= segment_gen_out_TYPE_A.squeeze(0)

								mdn_out_TYPE_A					= self.gen_state_fc2(segment_gen_out_TYPE_A)
								eos_TYPE_A						= mdn_out_TYPE_A[:,0:1]
								[mu1_TYPE_A, mu2_TYPE_A, sig1_TYPE_A, sig2_TYPE_A, rho_TYPE_A, pi_TYPE_A] = torch.split(mdn_out_TYPE_A[:,1:], self.num_mixtures, 1)
								sig1_TYPE_A						= sig1_TYPE_A.exp() + 1e-3
								sig2_TYPE_A						= sig2_TYPE_A.exp() + 1e-3
								rho_TYPE_A						= self.mdn_tanh(rho_TYPE_A)
								pi_TYPE_A						= self.mdn_softmax(pi_TYPE_A)
								term_out_TYPE_A					= self.term_fc1(segment_gen_out_TYPE_A)
								term_out_TYPE_A					= self.term_relu1(term_out_TYPE_A)
								term_out_TYPE_A					= self.term_fc2(term_out_TYPE_A)
								term_out_TYPE_A					= self.term_relu2(term_out_TYPE_A)
								term_out_TYPE_A					= self.term_fc3(term_out_TYPE_A)
								term_pred_TYPE_A				= self.term_sigmoid(term_out_TYPE_A)
								gaussian_TYPE_A					= gaussian_2d(segment_level_target_x, segment_level_target_y, mu1_TYPE_A, mu2_TYPE_A, sig1_TYPE_A, sig2_TYPE_A, rho_TYPE_A)
								loss_gaussian_TYPE_A			= - torch.log(torch.sum(pi_TYPE_A*gaussian_TYPE_A, dim=1) + 1e-5)

								TYPE_A_segment_term_loss		= self.term_bce_loss(term_out_TYPE_A.squeeze(1), segment_level_target_term)
								TYPE_A_segment_loc_loss			= torch.mean(loss_gaussian_TYPE_A)
								TYPE_A_segment_touch_loss		= self.mdn_bce_loss(eos_TYPE_A.squeeze(1), segment_level_target_eos)

								TYPE_A_segment_termination_loss.append(TYPE_A_segment_term_loss)
								TYPE_A_segment_loc_reconstruct_loss.append(TYPE_A_segment_loc_loss)
								TYPE_A_segment_touch_reconstruct_loss.append(TYPE_A_segment_touch_loss)

							# TYPE B
							if self.TYPE_B:
								unique_char_matrix_1			= segment_unique_char_matrices_1[segment_batch_id]
								unique_char_matrices_1			= torch.stack(unique_char_matrix_1)
								unique_char_matrices_1			= unique_char_matrices_1.squeeze(1)

								# segment_W_c_TYPE_B_RAW 			= torch.bmm(unique_char_matrices, segment_W_mean.repeat(unique_char_matrices.size(0), 1).unsqueeze(2)).squeeze(-1)
								segment_W_c_TYPE_B_RAW 			= torch.bmm(unique_char_matrices_1,
																segment_W_mean.repeat(unique_char_matrices_1.size(0), 1).unsqueeze(2)).squeeze(-1)
								segment_W_c_TYPE_B_RAW			= segment_W_c_TYPE_B_RAW.unsqueeze(0)

								segment_Wc_rec_TYPE_B, (c,h)	= self.magic_lstm(segment_W_c_TYPE_B_RAW)
								segment_Wc_rec_TYPE_B			= segment_Wc_rec_TYPE_B.squeeze(0)

								segment_Wcs_reconstruct_TYPE_B.append(segment_Wc_rec_TYPE_B)

								segment_W_lstm_in_TYPE_B		= []
								curr_id							= 0
								for i in range(user_segment_level_stroke_length[segment_batch_id][0]):
									segment_W_lstm_in_TYPE_B.append(segment_Wc_rec_TYPE_B[curr_id])
									if i in segment_SPLITS[segment_batch_id]:
										curr_id += 1
								segment_Wc_t_rec_TYPE_B			= torch.stack(segment_W_lstm_in_TYPE_B)

								segment_gen_lstm2_in_TYPE_B	= torch.cat([segment_level_gen_encoded, segment_Wc_t_rec_TYPE_B], -1)
								segment_gen_lstm2_in_TYPE_B 	= segment_gen_lstm2_in_TYPE_B.unsqueeze(0)
								segment_gen_out_TYPE_B, (c,h)	= self.gen_state_lstm2(segment_gen_lstm2_in_TYPE_B)
								segment_gen_out_TYPE_B			= segment_gen_out_TYPE_B.squeeze(0)

								mdn_out_TYPE_B					= self.gen_state_fc2(segment_gen_out_TYPE_B)
								eos_TYPE_B						= mdn_out_TYPE_B[:,0:1]
								[mu1_TYPE_B, mu2_TYPE_B, sig1_TYPE_B, sig2_TYPE_B, rho_TYPE_B, pi_TYPE_B] = torch.split(mdn_out_TYPE_B[:,1:], self.num_mixtures, 1)
								sig1_TYPE_B						= sig1_TYPE_B.exp() + 1e-3
								sig2_TYPE_B						= sig2_TYPE_B.exp() + 1e-3
								rho_TYPE_B						= self.mdn_tanh(rho_TYPE_B)
								pi_TYPE_B						= self.mdn_softmax(pi_TYPE_B)
								term_out_TYPE_B					= self.term_fc1(segment_gen_out_TYPE_B)
								term_out_TYPE_B					= self.term_relu1(term_out_TYPE_B)
								term_out_TYPE_B					= self.term_fc2(term_out_TYPE_B)
								term_out_TYPE_B					= self.term_relu2(term_out_TYPE_B)
								term_out_TYPE_B					= self.term_fc3(term_out_TYPE_B)
								term_pred_TYPE_B				= self.term_sigmoid(term_out_TYPE_B)
								gaussian_TYPE_B					= gaussian_2d(segment_level_target_x, segment_level_target_y, mu1_TYPE_B, mu2_TYPE_B, sig1_TYPE_B, sig2_TYPE_B, rho_TYPE_B)
								loss_gaussian_TYPE_B			= - torch.log(torch.sum(pi_TYPE_B*gaussian_TYPE_B, dim=1) + 1e-5)

								TYPE_B_segment_term_loss		= self.term_bce_loss(term_out_TYPE_B.squeeze(1), segment_level_target_term)
								TYPE_B_segment_loc_loss			= torch.mean(loss_gaussian_TYPE_B)
								TYPE_B_segment_touch_loss		= self.mdn_bce_loss(eos_TYPE_B.squeeze(1), segment_level_target_eos)

								TYPE_B_segment_termination_loss.append(TYPE_B_segment_term_loss)
								TYPE_B_segment_loc_reconstruct_loss.append(TYPE_B_segment_loc_loss)
								TYPE_B_segment_touch_reconstruct_loss.append(TYPE_B_segment_touch_loss)

						if self.ORIGINAL:
							ALL_ORIGINAL_segment_termination_loss.append(torch.mean(torch.stack(ORIGINAL_segment_termination_loss)))
							ALL_ORIGINAL_segment_loc_reconstruct_loss.append(torch.mean(torch.stack(ORIGINAL_segment_loc_reconstruct_loss)))
							ALL_ORIGINAL_segment_touch_reconstruct_loss.append(torch.mean(torch.stack(ORIGINAL_segment_touch_reconstruct_loss)))

						if self.TYPE_A:
							ALL_TYPE_A_segment_termination_loss.append(torch.mean(torch.stack(TYPE_A_segment_termination_loss)))
							ALL_TYPE_A_segment_loc_reconstruct_loss.append(torch.mean(torch.stack(TYPE_A_segment_loc_reconstruct_loss)))
							ALL_TYPE_A_segment_touch_reconstruct_loss.append(torch.mean(torch.stack(TYPE_A_segment_touch_reconstruct_loss)))

							if self.REC:
								TYPE_A_segment_WC_reconstruct_loss	= []
								for segment_batch_id in range(len(segment_Wc_rec_ORIGINAL)):
									segment_Wc_ORIGINAL	= segment_Wc_rec_ORIGINAL[segment_batch_id]
									segment_Wc_TYPE_A		= segment_Wcs_reconstruct_TYPE_A[segment_batch_id]
									segment_WC_reconstruct_loss_TYPE_A	= torch.mean(torch.mean(torch.mul(segment_Wc_ORIGINAL - segment_Wc_TYPE_A, segment_Wc_ORIGINAL - segment_Wc_TYPE_A), -1))
									TYPE_A_segment_WC_reconstruct_loss.append(segment_WC_reconstruct_loss_TYPE_A)
								ALL_TYPE_A_segment_WC_reconstruct_loss.append(torch.mean(torch.stack(TYPE_A_segment_WC_reconstruct_loss)))

						if self.TYPE_B:
							ALL_TYPE_B_segment_termination_loss.append(torch.mean(torch.stack(TYPE_B_segment_termination_loss)))
							ALL_TYPE_B_segment_loc_reconstruct_loss.append(torch.mean(torch.stack(TYPE_B_segment_loc_reconstruct_loss)))
							ALL_TYPE_B_segment_touch_reconstruct_loss.append(torch.mean(torch.stack(TYPE_B_segment_touch_reconstruct_loss)))

							if self.REC:
								TYPE_B_segment_WC_reconstruct_loss	= []
								for segment_batch_id in range(len(segment_Wc_rec_ORIGINAL)):
									segment_Wc_ORIGINAL	= segment_Wc_rec_ORIGINAL[segment_batch_id]
									segment_Wc_TYPE_B		= segment_Wcs_reconstruct_TYPE_B[segment_batch_id]
									segment_WC_reconstruct_loss_TYPE_B	= torch.mean(torch.mean(torch.mul(segment_Wc_ORIGINAL - segment_Wc_TYPE_B, segment_Wc_ORIGINAL - segment_Wc_TYPE_B), -1))
									TYPE_B_segment_WC_reconstruct_loss.append(segment_WC_reconstruct_loss_TYPE_B)
								ALL_TYPE_B_segment_WC_reconstruct_loss.append(torch.mean(torch.stack(TYPE_B_segment_WC_reconstruct_loss)))

					if self.TYPE_C:
						# target
						original_W_c	= word_Wc_rec_ORIGINAL[word_batch_id]
						word_Wc_rec_TYPE_C		= []
						for segment_batch_id in range(len(segment_Wc_rec_ORIGINAL)):
							if segment_batch_id == 0:
								for each_segment_Wc in segment_Wc_rec_ORIGINAL[segment_batch_id]:
									word_Wc_rec_TYPE_C.append(each_segment_Wc)
								prev_id = len(word_Wc_rec_TYPE_C) - 1
							else:
								prev_original_W_c	= original_W_c[prev_id]
								for each_segment_Wc in segment_Wc_rec_ORIGINAL[segment_batch_id]:
									magic_inp 	= torch.stack([prev_original_W_c, each_segment_Wc])
									magic_inp	= magic_inp.unsqueeze(0)
									type_c_out, (c,h) = self.magic_lstm(magic_inp)
									type_c_out = type_c_out.squeeze(0)
									word_Wc_rec_TYPE_C.append(type_c_out[-1])
								prev_id = len(word_Wc_rec_TYPE_C) - 1

						word_Wc_rec_TYPE_C	= torch.stack(word_Wc_rec_TYPE_C)
						word_Wcs_reconstruct_TYPE_C.append(word_Wc_rec_TYPE_C)

						if len(word_Wc_rec_TYPE_C) == len(word_SPLITS[word_batch_id]):
							word_W_lstm_in_TYPE_C		= []
							curr_id							= 0
							for i in range(user_word_level_stroke_length[word_batch_id][0]):
								word_W_lstm_in_TYPE_C.append(word_Wc_rec_TYPE_C[curr_id])
								if i in word_SPLITS[word_batch_id]:
									curr_id += 1
							word_Wc_t_rec_TYPE_C		= torch.stack(word_W_lstm_in_TYPE_C)

							word_gen_lstm2_in_TYPE_C	= torch.cat([word_level_gen_encoded, word_Wc_t_rec_TYPE_C], -1)
							word_gen_lstm2_in_TYPE_C 	= word_gen_lstm2_in_TYPE_C.unsqueeze(0)
							word_gen_out_TYPE_C, (c,h)	= self.gen_state_lstm2(word_gen_lstm2_in_TYPE_C)
							word_gen_out_TYPE_C			= word_gen_out_TYPE_C.squeeze(0)

							mdn_out_TYPE_C					= self.gen_state_fc2(word_gen_out_TYPE_C)
							eos_TYPE_C						= mdn_out_TYPE_C[:,0:1]
							[mu1_TYPE_C, mu2_TYPE_C, sig1_TYPE_C, sig2_TYPE_C, rho_TYPE_C, pi_TYPE_C] = torch.split(mdn_out_TYPE_C[:,1:], self.num_mixtures, 1)
							sig1_TYPE_C						= sig1_TYPE_C.exp() + 1e-3
							sig2_TYPE_C						= sig2_TYPE_C.exp() + 1e-3
							rho_TYPE_C						= self.mdn_tanh(rho_TYPE_C)
							pi_TYPE_C						= self.mdn_softmax(pi_TYPE_C)
							term_out_TYPE_C					= self.term_fc1(word_gen_out_TYPE_C)
							term_out_TYPE_C					= self.term_relu1(term_out_TYPE_C)
							term_out_TYPE_C					= self.term_fc2(term_out_TYPE_C)
							term_out_TYPE_C					= self.term_relu2(term_out_TYPE_C)
							term_out_TYPE_C					= self.term_fc3(term_out_TYPE_C)
							term_pred_TYPE_C				= self.term_sigmoid(term_out_TYPE_C)
							gaussian_TYPE_C					= gaussian_2d(word_level_target_x, word_level_target_y, mu1_TYPE_C, mu2_TYPE_C, sig1_TYPE_C, sig2_TYPE_C, rho_TYPE_C)
							loss_gaussian_TYPE_C			= - torch.log(torch.sum(pi_TYPE_C*gaussian_TYPE_C, dim=1) + 1e-5)

							TYPE_C_word_term_loss		= self.term_bce_loss(term_out_TYPE_C.squeeze(1), word_level_target_term)
							TYPE_C_word_loc_loss		= torch.mean(loss_gaussian_TYPE_C)
							TYPE_C_word_touch_loss		= self.mdn_bce_loss(eos_TYPE_C.squeeze(1), word_level_target_eos)

							TYPE_C_word_termination_loss.append(TYPE_C_word_term_loss)
							TYPE_C_word_loc_reconstruct_loss.append(TYPE_C_word_loc_loss)
							TYPE_C_word_touch_reconstruct_loss.append(TYPE_C_word_touch_loss)
						else:
							print ("not C")

					if self.TYPE_D:
						word_Wc_rec_TYPE_D		= []
						TYPE_D_REF				= []
						for segment_batch_id in range(len(segment_Wc_rec_ORIGINAL)):
							if segment_batch_id == 0:
								for each_segment_Wc in segment_Wc_rec_ORIGINAL[segment_batch_id]:
									word_Wc_rec_TYPE_D.append(each_segment_Wc)
								TYPE_D_REF.append(segment_Wc_rec_ORIGINAL[segment_batch_id][-1])
							else:
								for each_segment_Wc in segment_Wc_rec_ORIGINAL[segment_batch_id]:
									magic_inp 	= torch.cat([torch.stack(TYPE_D_REF, 0), each_segment_Wc.unsqueeze(0)], 0)
									magic_inp	= magic_inp.unsqueeze(0)
									TYPE_D_out, (c,h) = self.magic_lstm(magic_inp)
									TYPE_D_out = TYPE_D_out.squeeze(0)
									word_Wc_rec_TYPE_D.append(TYPE_D_out[-1])
								TYPE_D_REF.append(segment_Wc_rec_ORIGINAL[segment_batch_id][-1])
						word_Wc_rec_TYPE_D	= torch.stack(word_Wc_rec_TYPE_D)
						word_Wcs_reconstruct_TYPE_D.append(word_Wc_rec_TYPE_D)

						if len(word_Wc_rec_TYPE_D) == len(word_SPLITS[word_batch_id]):
							word_W_lstm_in_TYPE_D		= []
							curr_id						= 0
							for i in range(user_word_level_stroke_length[word_batch_id][0]):
								word_W_lstm_in_TYPE_D.append(word_Wc_rec_TYPE_D[curr_id])
								if i in word_SPLITS[word_batch_id]:
									curr_id += 1
							word_Wc_t_rec_TYPE_D		= torch.stack(word_W_lstm_in_TYPE_D)

							word_gen_lstm2_in_TYPE_D	= torch.cat([word_level_gen_encoded, word_Wc_t_rec_TYPE_D], -1)
							word_gen_lstm2_in_TYPE_D 	= word_gen_lstm2_in_TYPE_D.unsqueeze(0)
							word_gen_out_TYPE_D, (c,h)	= self.gen_state_lstm2(word_gen_lstm2_in_TYPE_D)
							word_gen_out_TYPE_D			= word_gen_out_TYPE_D.squeeze(0)

							mdn_out_TYPE_D					= self.gen_state_fc2(word_gen_out_TYPE_D)
							eos_TYPE_D						= mdn_out_TYPE_D[:,0:1]
							[mu1_TYPE_D, mu2_TYPE_D, sig1_TYPE_D, sig2_TYPE_D, rho_TYPE_D, pi_TYPE_D] = torch.split(mdn_out_TYPE_D[:,1:], self.num_mixtures, 1)
							sig1_TYPE_D						= sig1_TYPE_D.exp() + 1e-3
							sig2_TYPE_D						= sig2_TYPE_D.exp() + 1e-3
							rho_TYPE_D						= self.mdn_tanh(rho_TYPE_D)
							pi_TYPE_D						= self.mdn_softmax(pi_TYPE_D)
							term_out_TYPE_D					= self.term_fc1(word_gen_out_TYPE_D)
							term_out_TYPE_D					= self.term_relu1(term_out_TYPE_D)
							term_out_TYPE_D					= self.term_fc2(term_out_TYPE_D)
							term_out_TYPE_D					= self.term_relu2(term_out_TYPE_D)
							term_out_TYPE_D					= self.term_fc3(term_out_TYPE_D)
							term_pred_TYPE_D				= self.term_sigmoid(term_out_TYPE_D)
							gaussian_TYPE_D					= gaussian_2d(word_level_target_x, word_level_target_y, mu1_TYPE_D, mu2_TYPE_D, sig1_TYPE_D, sig2_TYPE_D, rho_TYPE_D)
							loss_gaussian_TYPE_D			= - torch.log(torch.sum(pi_TYPE_D*gaussian_TYPE_D, dim=1) + 1e-5)

							TYPE_D_word_term_loss		= self.term_bce_loss(term_out_TYPE_D.squeeze(1), word_level_target_term)
							TYPE_D_word_loc_loss		= torch.mean(loss_gaussian_TYPE_D)
							TYPE_D_word_touch_loss		= self.mdn_bce_loss(eos_TYPE_D.squeeze(1), word_level_target_eos)

							TYPE_D_word_termination_loss.append(TYPE_D_word_term_loss)
							TYPE_D_word_loc_reconstruct_loss.append(TYPE_D_word_loc_loss)
							TYPE_D_word_touch_reconstruct_loss.append(TYPE_D_word_touch_loss)
						else:
							print ("not D")

				# word
				if self.ORIGINAL:
					ALL_ORIGINAL_word_termination_loss.append(torch.mean(torch.stack(ORIGINAL_word_termination_loss)))
					ALL_ORIGINAL_word_loc_reconstruct_loss.append(torch.mean(torch.stack(ORIGINAL_word_loc_reconstruct_loss)))
					ALL_ORIGINAL_word_touch_reconstruct_loss.append(torch.mean(torch.stack(ORIGINAL_word_touch_reconstruct_loss)))

				if self.TYPE_A:
					ALL_TYPE_A_word_termination_loss.append(torch.mean(torch.stack(TYPE_A_word_termination_loss)))
					ALL_TYPE_A_word_loc_reconstruct_loss.append(torch.mean(torch.stack(TYPE_A_word_loc_reconstruct_loss)))
					ALL_TYPE_A_word_touch_reconstruct_loss.append(torch.mean(torch.stack(TYPE_A_word_touch_reconstruct_loss)))

					if self.REC:
						TYPE_A_word_WC_reconstruct_loss	= []
						for word_batch_id in range(len(word_Wc_rec_ORIGINAL)):
							word_Wc_ORIGINAL				= word_Wc_rec_ORIGINAL[word_batch_id]
							word_Wc_TYPE_A					= word_Wcs_reconstruct_TYPE_A[word_batch_id]
							if len(word_Wc_ORIGINAL) == len(word_Wc_TYPE_A):
								word_WC_reconstruct_loss_TYPE_A	= torch.mean(torch.mean(torch.mul(word_Wc_ORIGINAL - word_Wc_TYPE_A, word_Wc_ORIGINAL - word_Wc_TYPE_A), -1))
								TYPE_A_word_WC_reconstruct_loss.append(word_WC_reconstruct_loss_TYPE_A)
						if len(TYPE_A_word_WC_reconstruct_loss) > 0:
							ALL_TYPE_A_word_WC_reconstruct_loss.append(torch.mean(torch.stack(TYPE_A_word_WC_reconstruct_loss)))

				if self.TYPE_B:
					ALL_TYPE_B_word_termination_loss.append(torch.mean(torch.stack(TYPE_B_word_termination_loss)))
					ALL_TYPE_B_word_loc_reconstruct_loss.append(torch.mean(torch.stack(TYPE_B_word_loc_reconstruct_loss)))
					ALL_TYPE_B_word_touch_reconstruct_loss.append(torch.mean(torch.stack(TYPE_B_word_touch_reconstruct_loss)))

					if self.REC:
						TYPE_B_word_WC_reconstruct_loss	= []
						for word_batch_id in range(len(word_Wc_rec_ORIGINAL)):
							word_Wc_ORIGINAL				= word_Wc_rec_ORIGINAL[word_batch_id]
							word_Wc_TYPE_B					= word_Wcs_reconstruct_TYPE_B[word_batch_id]
							if len(word_Wc_ORIGINAL) == len(word_Wc_TYPE_B):
								word_WC_reconstruct_loss_TYPE_B	= torch.mean(torch.mean(torch.mul(word_Wc_ORIGINAL - word_Wc_TYPE_B, word_Wc_ORIGINAL - word_Wc_TYPE_B), -1))
								TYPE_B_word_WC_reconstruct_loss.append(word_WC_reconstruct_loss_TYPE_B)
						if len(TYPE_B_word_WC_reconstruct_loss) > 0:
							ALL_TYPE_B_word_WC_reconstruct_loss.append(torch.mean(torch.stack(TYPE_B_word_WC_reconstruct_loss)))

				if self.TYPE_C:
					ALL_TYPE_C_word_termination_loss.append(torch.mean(torch.stack(TYPE_C_word_termination_loss)))
					ALL_TYPE_C_word_loc_reconstruct_loss.append(torch.mean(torch.stack(TYPE_C_word_loc_reconstruct_loss)))
					ALL_TYPE_C_word_touch_reconstruct_loss.append(torch.mean(torch.stack(TYPE_C_word_touch_reconstruct_loss)))

					if self.REC:
						TYPE_C_word_WC_reconstruct_loss	= []
						for word_batch_id in range(len(word_Wc_rec_ORIGINAL)):
							word_Wc_ORIGINAL				= word_Wc_rec_ORIGINAL[word_batch_id]
							word_Wc_TYPE_C					= word_Wcs_reconstruct_TYPE_C[word_batch_id]
							if len(word_Wc_ORIGINAL) == len(word_Wc_TYPE_C):
								word_WC_reconstruct_loss_TYPE_C	= torch.mean(torch.mean(torch.mul(word_Wc_ORIGINAL - word_Wc_TYPE_C, word_Wc_ORIGINAL - word_Wc_TYPE_C), -1))
								TYPE_C_word_WC_reconstruct_loss.append(word_WC_reconstruct_loss_TYPE_C)
						if len(TYPE_C_word_WC_reconstruct_loss) > 0:
							ALL_TYPE_C_word_WC_reconstruct_loss.append(torch.mean(torch.stack(TYPE_C_word_WC_reconstruct_loss)))

				if self.TYPE_D:
					ALL_TYPE_D_word_termination_loss.append(torch.mean(torch.stack(TYPE_D_word_termination_loss)))
					ALL_TYPE_D_word_loc_reconstruct_loss.append(torch.mean(torch.stack(TYPE_D_word_loc_reconstruct_loss)))
					ALL_TYPE_D_word_touch_reconstruct_loss.append(torch.mean(torch.stack(TYPE_D_word_touch_reconstruct_loss)))

					if self.REC:
						TYPE_D_word_WC_reconstruct_loss	= []
						for word_batch_id in range(len(word_Wc_rec_ORIGINAL)):
							word_Wc_ORIGINAL				= word_Wc_rec_ORIGINAL[word_batch_id]
							word_Wc_TYPE_D					= word_Wcs_reconstruct_TYPE_D[word_batch_id]
							if len(word_Wc_ORIGINAL) == len(word_Wc_TYPE_D):
								word_WC_reconstruct_loss_TYPE_D	= torch.mean(torch.mean(torch.mul(word_Wc_ORIGINAL - word_Wc_TYPE_D, word_Wc_ORIGINAL - word_Wc_TYPE_D), -1))
								TYPE_D_word_WC_reconstruct_loss.append(word_WC_reconstruct_loss_TYPE_D)
						if len(TYPE_D_word_WC_reconstruct_loss) > 0:
							ALL_TYPE_D_word_WC_reconstruct_loss.append(torch.mean(torch.stack(TYPE_D_word_WC_reconstruct_loss)))

				# segment
				if self.segment_loss:
					SUPER_ALL_segment_W_consistency_loss.append(torch.mean(torch.stack(ALL_segment_W_consistency_loss)))

					if self.ORIGINAL:
						SUPER_ALL_ORIGINAL_segment_termination_loss.append(torch.mean(torch.stack(ALL_ORIGINAL_segment_termination_loss)))
						SUPER_ALL_ORIGINAL_segment_loc_reconstruct_loss.append(torch.mean(torch.stack(ALL_ORIGINAL_segment_loc_reconstruct_loss)))
						SUPER_ALL_ORIGINAL_segment_touch_reconstruct_loss.append(torch.mean(torch.stack(ALL_ORIGINAL_segment_touch_reconstruct_loss)))

					if self.TYPE_A:
						SUPER_ALL_TYPE_A_segment_termination_loss.append(torch.mean(torch.stack(ALL_TYPE_A_segment_termination_loss)))
						SUPER_ALL_TYPE_A_segment_loc_reconstruct_loss.append(torch.mean(torch.stack(ALL_TYPE_A_segment_loc_reconstruct_loss)))
						SUPER_ALL_TYPE_A_segment_touch_reconstruct_loss.append(torch.mean(torch.stack(ALL_TYPE_A_segment_touch_reconstruct_loss)))
						if self.REC:
							SUPER_ALL_TYPE_A_segment_WC_reconstruct_loss.append(torch.mean(torch.stack(ALL_TYPE_A_segment_WC_reconstruct_loss)))

					if self.TYPE_B:
						SUPER_ALL_TYPE_B_segment_termination_loss.append(torch.mean(torch.stack(ALL_TYPE_B_segment_termination_loss)))
						SUPER_ALL_TYPE_B_segment_loc_reconstruct_loss.append(torch.mean(torch.stack(ALL_TYPE_B_segment_loc_reconstruct_loss)))
						SUPER_ALL_TYPE_B_segment_touch_reconstruct_loss.append(torch.mean(torch.stack(ALL_TYPE_B_segment_touch_reconstruct_loss)))
						if self.REC:
							SUPER_ALL_TYPE_B_segment_WC_reconstruct_loss.append(torch.mean(torch.stack(ALL_TYPE_B_segment_WC_reconstruct_loss)))

		total_sentence_loss	= 0
		sentence_losses = []
		if self.sentence_loss:
			mean_ORIGINAL_sentence_termination_loss = 0
			mean_ORIGINAL_sentence_loc_reconstruct_loss = 0
			mean_ORIGINAL_sentence_touch_reconstruct_loss = 0
			mean_TYPE_A_sentence_termination_loss = 0
			mean_TYPE_A_sentence_loc_reconstruct_loss = 0
			mean_TYPE_A_sentence_touch_reconstruct_loss = 0
			mean_TYPE_B_sentence_termination_loss = 0
			mean_TYPE_B_sentence_loc_reconstruct_loss = 0
			mean_TYPE_B_sentence_touch_reconstruct_loss = 0
			mean_TYPE_A_sentence_WC_reconstruct_loss = 0
			mean_TYPE_B_sentence_WC_reconstruct_loss = 0

			mean_sentence_W_consistency_loss 				= torch.mean(torch.stack(ALL_sentence_W_consistency_loss))
			if self.ORIGINAL:
				mean_ORIGINAL_sentence_termination_loss 		= torch.mean(torch.stack(ALL_ORIGINAL_sentence_termination_loss))
				mean_ORIGINAL_sentence_loc_reconstruct_loss 	= torch.mean(torch.stack(ALL_ORIGINAL_sentence_loc_reconstruct_loss))
				mean_ORIGINAL_sentence_touch_reconstruct_loss 	= torch.mean(torch.stack(ALL_ORIGINAL_sentence_touch_reconstruct_loss))
			if self.TYPE_A:
				mean_TYPE_A_sentence_termination_loss 			= torch.mean(torch.stack(ALL_TYPE_A_sentence_termination_loss))
				mean_TYPE_A_sentence_loc_reconstruct_loss		= torch.mean(torch.stack(ALL_TYPE_A_sentence_loc_reconstruct_loss))
				mean_TYPE_A_sentence_touch_reconstruct_loss 	= torch.mean(torch.stack(ALL_TYPE_A_sentence_touch_reconstruct_loss))
				if self.REC:
					mean_TYPE_A_sentence_WC_reconstruct_loss 		= torch.mean(torch.stack(ALL_TYPE_A_sentence_WC_reconstruct_loss))
			if self.TYPE_B:
				mean_TYPE_B_sentence_termination_loss 			= torch.mean(torch.stack(ALL_TYPE_B_sentence_termination_loss))
				mean_TYPE_B_sentence_loc_reconstruct_loss 		= torch.mean(torch.stack(ALL_TYPE_B_sentence_loc_reconstruct_loss))
				mean_TYPE_B_sentence_touch_reconstruct_loss 	= torch.mean(torch.stack(ALL_TYPE_B_sentence_touch_reconstruct_loss))
				if self.REC:
					mean_TYPE_B_sentence_WC_reconstruct_loss		= torch.mean(torch.stack(ALL_TYPE_B_sentence_WC_reconstruct_loss))

			total_sentence_loss = mean_sentence_W_consistency_loss + mean_ORIGINAL_sentence_termination_loss + mean_ORIGINAL_sentence_loc_reconstruct_loss + mean_ORIGINAL_sentence_touch_reconstruct_loss + mean_TYPE_A_sentence_termination_loss + mean_TYPE_A_sentence_loc_reconstruct_loss + mean_TYPE_A_sentence_touch_reconstruct_loss + mean_TYPE_B_sentence_termination_loss + mean_TYPE_B_sentence_loc_reconstruct_loss + mean_TYPE_B_sentence_touch_reconstruct_loss + mean_TYPE_A_sentence_WC_reconstruct_loss + mean_TYPE_B_sentence_WC_reconstruct_loss
			sentence_losses = [total_sentence_loss, mean_sentence_W_consistency_loss, mean_ORIGINAL_sentence_termination_loss, mean_ORIGINAL_sentence_loc_reconstruct_loss, mean_ORIGINAL_sentence_touch_reconstruct_loss, mean_TYPE_A_sentence_termination_loss, mean_TYPE_A_sentence_loc_reconstruct_loss, mean_TYPE_A_sentence_touch_reconstruct_loss, mean_TYPE_B_sentence_termination_loss, mean_TYPE_B_sentence_loc_reconstruct_loss, mean_TYPE_B_sentence_touch_reconstruct_loss, mean_TYPE_A_sentence_WC_reconstruct_loss, mean_TYPE_B_sentence_WC_reconstruct_loss]

		total_word_loss	= 0
		word_losses = []
		if self.word_loss:
			mean_ORIGINAL_word_termination_loss 				= 0
			mean_ORIGINAL_word_loc_reconstruct_loss 			= 0
			mean_ORIGINAL_word_touch_reconstruct_loss 			= 0
			mean_TYPE_A_word_termination_loss 					= 0
			mean_TYPE_A_word_loc_reconstruct_loss 				= 0
			mean_TYPE_A_word_touch_reconstruct_loss 			= 0
			mean_TYPE_B_word_termination_loss 					= 0
			mean_TYPE_B_word_loc_reconstruct_loss 				= 0
			mean_TYPE_B_word_touch_reconstruct_loss 			= 0
			mean_TYPE_C_word_termination_loss 					= 0
			mean_TYPE_C_word_loc_reconstruct_loss 				= 0
			mean_TYPE_C_word_touch_reconstruct_loss 			= 0
			mean_TYPE_D_word_termination_loss 					= 0
			mean_TYPE_D_word_loc_reconstruct_loss 				= 0
			mean_TYPE_D_word_touch_reconstruct_loss 			= 0
			mean_TYPE_A_word_WC_reconstruct_loss 				= 0
			mean_TYPE_B_word_WC_reconstruct_loss 				= 0
			mean_TYPE_C_word_WC_reconstruct_loss 				= 0
			mean_TYPE_D_word_WC_reconstruct_loss 				= 0

			mean_word_W_consistency_loss 						= torch.mean(torch.stack(ALL_word_W_consistency_loss))
			if self.ORIGINAL:
				mean_ORIGINAL_word_termination_loss 			= torch.mean(torch.stack(ALL_ORIGINAL_word_termination_loss))
				mean_ORIGINAL_word_loc_reconstruct_loss 		= torch.mean(torch.stack(ALL_ORIGINAL_word_loc_reconstruct_loss))
				mean_ORIGINAL_word_touch_reconstruct_loss 		= torch.mean(torch.stack(ALL_ORIGINAL_word_touch_reconstruct_loss))
			if self.TYPE_A:
				mean_TYPE_A_word_termination_loss		 		= torch.mean(torch.stack(ALL_TYPE_A_word_termination_loss))
				mean_TYPE_A_word_loc_reconstruct_loss 			= torch.mean(torch.stack(ALL_TYPE_A_word_loc_reconstruct_loss))
				mean_TYPE_A_word_touch_reconstruct_loss 		= torch.mean(torch.stack(ALL_TYPE_A_word_touch_reconstruct_loss))
				if self.REC:
					mean_TYPE_A_word_WC_reconstruct_loss 			= torch.mean(torch.stack(ALL_TYPE_A_word_WC_reconstruct_loss))
			if self.TYPE_B:
				mean_TYPE_B_word_termination_loss 				= torch.mean(torch.stack(ALL_TYPE_B_word_termination_loss))
				mean_TYPE_B_word_loc_reconstruct_loss 			= torch.mean(torch.stack(ALL_TYPE_B_word_loc_reconstruct_loss))
				mean_TYPE_B_word_touch_reconstruct_loss 		= torch.mean(torch.stack(ALL_TYPE_B_word_touch_reconstruct_loss))
				if self.REC:
					mean_TYPE_B_word_WC_reconstruct_loss 			= torch.mean(torch.stack(ALL_TYPE_B_word_WC_reconstruct_loss))
			if self.TYPE_C:
				mean_TYPE_C_word_termination_loss 				= torch.mean(torch.stack(ALL_TYPE_C_word_termination_loss))
				mean_TYPE_C_word_loc_reconstruct_loss 			= torch.mean(torch.stack(ALL_TYPE_C_word_loc_reconstruct_loss))
				mean_TYPE_C_word_touch_reconstruct_loss 		= torch.mean(torch.stack(ALL_TYPE_C_word_touch_reconstruct_loss))
				if self.REC:
					mean_TYPE_C_word_WC_reconstruct_loss 			= torch.mean(torch.stack(ALL_TYPE_C_word_WC_reconstruct_loss))
			if self.TYPE_D:
				mean_TYPE_D_word_termination_loss 				= torch.mean(torch.stack(ALL_TYPE_D_word_termination_loss))
				mean_TYPE_D_word_loc_reconstruct_loss 			= torch.mean(torch.stack(ALL_TYPE_D_word_loc_reconstruct_loss))
				mean_TYPE_D_word_touch_reconstruct_loss 		= torch.mean(torch.stack(ALL_TYPE_D_word_touch_reconstruct_loss))
				if self.REC:
					mean_TYPE_D_word_WC_reconstruct_loss 			= torch.mean(torch.stack(ALL_TYPE_D_word_WC_reconstruct_loss))

			total_word_loss = mean_word_W_consistency_loss + mean_ORIGINAL_word_termination_loss + mean_ORIGINAL_word_loc_reconstruct_loss + mean_ORIGINAL_word_touch_reconstruct_loss + mean_TYPE_A_word_termination_loss + mean_TYPE_A_word_loc_reconstruct_loss + mean_TYPE_A_word_touch_reconstruct_loss + mean_TYPE_B_word_termination_loss + mean_TYPE_B_word_loc_reconstruct_loss + mean_TYPE_B_word_touch_reconstruct_loss + mean_TYPE_C_word_termination_loss + mean_TYPE_C_word_loc_reconstruct_loss + mean_TYPE_C_word_touch_reconstruct_loss + mean_TYPE_D_word_termination_loss + mean_TYPE_D_word_loc_reconstruct_loss + mean_TYPE_D_word_touch_reconstruct_loss + mean_TYPE_A_word_WC_reconstruct_loss + mean_TYPE_B_word_WC_reconstruct_loss + mean_TYPE_C_word_WC_reconstruct_loss + mean_TYPE_D_word_WC_reconstruct_loss
			word_losses = [total_word_loss, mean_word_W_consistency_loss, mean_ORIGINAL_word_termination_loss, mean_ORIGINAL_word_loc_reconstruct_loss, mean_ORIGINAL_word_touch_reconstruct_loss, mean_TYPE_A_word_termination_loss, mean_TYPE_A_word_loc_reconstruct_loss, mean_TYPE_A_word_touch_reconstruct_loss, mean_TYPE_B_word_termination_loss, mean_TYPE_B_word_loc_reconstruct_loss, mean_TYPE_B_word_touch_reconstruct_loss, mean_TYPE_C_word_termination_loss, mean_TYPE_C_word_loc_reconstruct_loss, mean_TYPE_C_word_touch_reconstruct_loss, mean_TYPE_D_word_termination_loss, mean_TYPE_D_word_loc_reconstruct_loss, mean_TYPE_D_word_touch_reconstruct_loss, mean_TYPE_A_word_WC_reconstruct_loss, mean_TYPE_B_word_WC_reconstruct_loss, mean_TYPE_C_word_WC_reconstruct_loss, mean_TYPE_D_word_WC_reconstruct_loss]

		total_segment_loss = 0
		segment_losses = []
		if self.segment_loss:
			mean_segment_W_consistency_loss = torch.mean(torch.stack(SUPER_ALL_segment_W_consistency_loss))

			mean_ORIGINAL_segment_termination_loss = 0
			mean_ORIGINAL_segment_loc_reconstruct_loss = 0
			mean_ORIGINAL_segment_touch_reconstruct_loss = 0
			mean_TYPE_A_segment_termination_loss = 0
			mean_TYPE_A_segment_loc_reconstruct_loss = 0
			mean_TYPE_A_segment_touch_reconstruct_loss = 0
			mean_TYPE_B_segment_termination_loss = 0
			mean_TYPE_B_segment_loc_reconstruct_loss = 0
			mean_TYPE_B_segment_touch_reconstruct_loss = 0
			mean_TYPE_A_segment_WC_reconstruct_loss = 0
			mean_TYPE_B_segment_WC_reconstruct_loss = 0

			if self.ORIGINAL:
				mean_ORIGINAL_segment_termination_loss = torch.mean(torch.stack(SUPER_ALL_ORIGINAL_segment_termination_loss))
				mean_ORIGINAL_segment_loc_reconstruct_loss = torch.mean(torch.stack(SUPER_ALL_ORIGINAL_segment_loc_reconstruct_loss))
				mean_ORIGINAL_segment_touch_reconstruct_loss = torch.mean(torch.stack(SUPER_ALL_ORIGINAL_segment_touch_reconstruct_loss))
			if self.TYPE_A:
				mean_TYPE_A_segment_termination_loss = torch.mean(torch.stack(SUPER_ALL_TYPE_A_segment_termination_loss))
				mean_TYPE_A_segment_loc_reconstruct_loss = torch.mean(torch.stack(SUPER_ALL_TYPE_A_segment_loc_reconstruct_loss))
				mean_TYPE_A_segment_touch_reconstruct_loss = torch.mean(torch.stack(SUPER_ALL_TYPE_A_segment_touch_reconstruct_loss))
				if self.REC:
					mean_TYPE_A_segment_WC_reconstruct_loss = torch.mean(torch.stack(SUPER_ALL_TYPE_A_segment_WC_reconstruct_loss))
			if self.TYPE_B:
				mean_TYPE_B_segment_termination_loss = torch.mean(torch.stack(SUPER_ALL_TYPE_B_segment_termination_loss))
				mean_TYPE_B_segment_loc_reconstruct_loss = torch.mean(torch.stack(SUPER_ALL_TYPE_B_segment_loc_reconstruct_loss))
				mean_TYPE_B_segment_touch_reconstruct_loss = torch.mean(torch.stack(SUPER_ALL_TYPE_B_segment_touch_reconstruct_loss))
				if self.REC:
					mean_TYPE_B_segment_WC_reconstruct_loss = torch.mean(torch.stack(SUPER_ALL_TYPE_B_segment_WC_reconstruct_loss))

			total_segment_loss = mean_segment_W_consistency_loss + mean_ORIGINAL_segment_termination_loss + mean_ORIGINAL_segment_loc_reconstruct_loss + mean_ORIGINAL_segment_touch_reconstruct_loss + mean_TYPE_A_segment_termination_loss + mean_TYPE_A_segment_loc_reconstruct_loss + mean_TYPE_A_segment_touch_reconstruct_loss + mean_TYPE_B_segment_termination_loss + mean_TYPE_B_segment_loc_reconstruct_loss + mean_TYPE_B_segment_touch_reconstruct_loss + mean_TYPE_A_segment_WC_reconstruct_loss + mean_TYPE_B_segment_WC_reconstruct_loss
			segment_losses = [total_segment_loss, mean_segment_W_consistency_loss, mean_ORIGINAL_segment_termination_loss, mean_ORIGINAL_segment_loc_reconstruct_loss, mean_ORIGINAL_segment_touch_reconstruct_loss, mean_TYPE_A_segment_termination_loss, mean_TYPE_A_segment_loc_reconstruct_loss, mean_TYPE_A_segment_touch_reconstruct_loss, mean_TYPE_B_segment_termination_loss, mean_TYPE_B_segment_loc_reconstruct_loss, mean_TYPE_B_segment_touch_reconstruct_loss, mean_TYPE_A_segment_WC_reconstruct_loss, mean_TYPE_B_segment_WC_reconstruct_loss]

		total_loss			= total_sentence_loss + total_word_loss + total_segment_loss

		return total_loss, sentence_losses, word_losses, segment_losses

	def sample(self, inputs):
		[	word_level_stroke_in, word_level_stroke_out, word_level_stroke_length,
			word_level_term, word_level_char, word_level_char_length, segment_level_stroke_in,
			segment_level_stroke_out, segment_level_stroke_length, segment_level_term,
			segment_level_char, segment_level_char_length	] = inputs

		word_inf_state_out 			= self.inf_state_fc1(word_level_stroke_out[0])
		word_inf_state_out			= self.inf_state_relu(word_inf_state_out)
		word_inf_state_out, (c,h) 	= self.inf_state_lstm(word_inf_state_out)

		user_word_level_char		= word_level_char[0]
		user_word_level_term		= word_level_term[0]

		raw_Ws			= []
		original_Wc		= []

		word_batch_id = 0

		# ORIGINAL
		curr_seq_len 	= word_level_stroke_length[0][word_batch_id][0]
		curr_char_len	= word_level_char_length[0][word_batch_id][0]

		char_vector				= torch.eye(len(CHARACTERS))[user_word_level_char[word_batch_id][:curr_char_len]].to(self.device)
		current_term			= user_word_level_term[word_batch_id][:curr_seq_len].unsqueeze(-1)
		split_ids				= torch.nonzero(current_term)[:,0]

		# char_vector				= self.char_vec_fc(char_vector)
		# char_vector				= self.char_vec_relu(char_vector)
		char_vector_1				= self.char_vec_fc_1(char_vector)
		char_vector_1				= self.char_vec_relu_1(char_vector_1)

		# unique_char_matrices	= []
		# for cid in range(len(char_vector)):
		# 	unique_char_vector		= char_vector[cid:cid+1]
		# 	unique_char_out			= unique_char_vector.unsqueeze(0)
		# 	unique_char_out, (c,h)	= self.char_lstm(unique_char_out)
		# 	unique_char_out			= unique_char_out.squeeze(0)
		# 	unique_char_out			= self.char_vec_fc2(unique_char_out)
		# 	unique_char_matrix		= unique_char_out.view([-1,1,self.weight_dim,self.weight_dim])
		# 	unique_char_matrix		= unique_char_matrix.squeeze(1)
		# 	unique_char_matrices.append(unique_char_matrix)

		unique_char_matrices_1			= []
		for cid in range(len(char_vector)):
			# Tower 1
			unique_char_vector_1		= char_vector_1[cid:cid+1]
			unique_char_input_1			= unique_char_vector_1.unsqueeze(0)
			unique_char_out_1, (c,h)	= self.char_lstm_1(unique_char_input_1)
			unique_char_out_1			= unique_char_out_1.squeeze(0)
			unique_char_out_1			= self.char_vec_fc2_1(unique_char_out_1)
			unique_char_matrix_1		= unique_char_out_1.view([-1,1,self.weight_dim,self.weight_dim])
			unique_char_matrix_1		= unique_char_matrix_1.squeeze(1)
			unique_char_matrices_1.append(unique_char_matrix_1)

		# Tower 1
		char_out_1				= char_vector_1.unsqueeze(0)
		char_out_1, (c,h) 		= self.char_lstm_1(char_out_1)
		char_out_1 				= char_out_1.squeeze(0)
		char_out_1				= self.char_vec_fc2_1(char_out_1)
		char_matrix_1			= char_out_1.view([-1,1,self.weight_dim,self.weight_dim])
		char_matrix_1			= char_matrix_1.squeeze(1)
		char_matrix_inv_1		= torch.inverse(char_matrix_1)

		W_c_t					= word_inf_state_out[word_batch_id][:curr_seq_len]
		W_c						= torch.stack([W_c_t[i] for i in split_ids])
		original_Wc.append(W_c)

		W						= torch.bmm(char_matrix_inv_1,
											  W_c.unsqueeze(2)).squeeze(-1)

		user_segment_level_stroke_length	= segment_level_stroke_length[0][word_batch_id]
		user_segment_level_char_length		= segment_level_char_length[0][word_batch_id]
		user_segment_level_term				= segment_level_term[0][word_batch_id]
		user_segment_level_char				= segment_level_char[0][word_batch_id]
		user_segment_level_stroke_in		= segment_level_stroke_in[0][word_batch_id]
		user_segment_level_stroke_out		= segment_level_stroke_out[0][word_batch_id]

		segment_inf_state_out				= self.inf_state_fc1(user_segment_level_stroke_out)
		segment_inf_state_out				= self.inf_state_relu(segment_inf_state_out)
		segment_inf_state_out, (c,h)		= self.inf_state_lstm(segment_inf_state_out)

		segment_W_c = []
		for segment_batch_id in range(len(user_segment_level_char)):
			curr_seq_len			= user_segment_level_stroke_length[segment_batch_id][0]
			curr_char_len			= user_segment_level_char_length[segment_batch_id][0]
			current_term			= user_segment_level_term[segment_batch_id][:curr_seq_len].unsqueeze(-1)
			split_ids				= torch.nonzero(current_term)[:,0]

			seg_W_c_t				= segment_inf_state_out[segment_batch_id][:curr_seq_len]
			seg_W_c					= torch.stack([seg_W_c_t[i] for i in split_ids])
			segment_W_c.append(seg_W_c)

		target_characters_ids = word_level_char[0][0][:word_level_char_length[0][0]]
		target_characters = ''.join([CHARACTERS[i] for i in target_characters_ids])

		mean_global_W	= torch.mean(W, 0)

		TYPE_A_WC	= torch.bmm(char_matrix_1,
								  mean_global_W.repeat(char_matrix_1.size(0), 1).unsqueeze(2)).squeeze(-1)

		unique_char_matrix_1 = torch.stack(unique_char_matrices_1)
		unique_char_matrix_1 = unique_char_matrix_1.squeeze(1)

		TYPE_B_WC_RAW	= torch.bmm(unique_char_matrix_1,
									  mean_global_W.repeat(unique_char_matrix_1.size(0), 1).unsqueeze(2)).squeeze(-1)

		TYPE_B_WC_RAW	= TYPE_B_WC_RAW.unsqueeze(0)
		TYPE_B_WC, (c,h)	= self.magic_lstm(TYPE_B_WC_RAW)
		TYPE_B_WC = TYPE_B_WC.squeeze(0)

		# CC
		TYPE_C_WC		= []
		for segment_batch_id in range(len(segment_W_c)):
			if segment_batch_id == 0:
				for each_segment_Wc in segment_W_c[segment_batch_id]:
					TYPE_C_WC.append(each_segment_Wc)
				prev_id = len(TYPE_C_WC) - 1
			else:
				prev_original_W_c	= W_c[prev_id]
				for each_segment_Wc in segment_W_c[segment_batch_id]:
					magic_inp 	= torch.stack([prev_original_W_c, each_segment_Wc])
					magic_inp	= magic_inp.unsqueeze(0)
					type_c_out, (c,h) = self.magic_lstm(magic_inp)
					type_c_out = type_c_out.squeeze(0)
					TYPE_C_WC.append(type_c_out[-1])
				prev_id = len(TYPE_C_WC) - 1
		TYPE_C_WC	= torch.stack(TYPE_C_WC)


		# DD
		TYPE_D_WC 		= []
		TYPE_D_REF		= []
		for segment_batch_id in range(len(segment_W_c)):
			if segment_batch_id == 0:
				for each_segment_Wc in segment_W_c[segment_batch_id]:
					TYPE_D_WC.append(each_segment_Wc)
				TYPE_D_REF.append(segment_W_c[segment_batch_id][-1])
			else:
				for each_segment_Wc in segment_W_c[segment_batch_id]:
					magic_inp 	= torch.cat([torch.stack(TYPE_D_REF, 0), each_segment_Wc.unsqueeze(0)], 0)
					magic_inp	= magic_inp.unsqueeze(0)
					TYPE_D_out, (c,h) = self.magic_lstm(magic_inp)
					TYPE_D_out = TYPE_D_out.squeeze(0)
					TYPE_D_WC.append(TYPE_D_out[-1])
				TYPE_D_REF.append(segment_W_c[segment_batch_id][-1])
		TYPE_D_WC	= torch.stack(TYPE_D_WC)


		o_tc = ''.join([CHARACTERS[c] for c in word_level_char[0][0][:word_level_char_length[0][0]]])
		o_commands = self.sample_from_w(original_Wc[0], o_tc)
		if len(TYPE_A_WC) == len(original_Wc[0]):
			a_commands = self.sample_from_w(TYPE_A_WC, target_characters)
		else:
			a_commands = [[0,0,0]]

		if len(TYPE_B_WC) == len(original_Wc[0]):
			b_commands = self.sample_from_w(TYPE_B_WC, target_characters)
		else:
			b_commands = [[0,0,0]]

		if len(TYPE_C_WC) == len(original_Wc[0]):
			c_commands = self.sample_from_w(TYPE_C_WC, target_characters)
		else:
			c_commands = [[0,0,0]]

		if len(TYPE_D_WC) == len(original_Wc[0]):
			d_commands = self.sample_from_w(TYPE_D_WC, target_characters)
		else:
			d_commands = [[0,0,0]]

		return [word_level_stroke_out[0][0], o_commands, a_commands, b_commands, c_commands, d_commands]

	def sample_from_w(self, W_c_rec, target_sentence):
		gen_input = torch.zeros([1, 1, 3]).to(self.device)
		current_char_id_count = 0

		gc1 = torch.zeros([self.num_layers, 1, self.weight_dim]).to(self.device)
		gh1 = torch.zeros([self.num_layers, 1, self.weight_dim]).to(self.device)
		gc2 = torch.zeros([self.num_layers, 1, self.weight_dim * 2]).to(self.device)
		gh2 = torch.zeros([self.num_layers, 1, self.weight_dim * 2]).to(self.device)

		terms = []
		commands = []
		character_nums = 0
		cx, cy = 100, 150
		for zz in range(800):
			W_c_t_now = W_c_rec[current_char_id_count:current_char_id_count + 1]

			gen_state = self.gen_state_fc1(gen_input)
			gen_state = self.gen_state_relu(gen_state)
			gen_state, (gc1, gh1) = self.gen_state_lstm1(gen_state, (gc1, gh1))
			gen_encoded = gen_state.squeeze(0)

			gen_lstm2_input = torch.cat([gen_encoded, W_c_t_now], -1)
			gen_lstm2_input = gen_lstm2_input.view([1, 1, self.weight_dim * 2])
			gen_out, (gc2, gh2) = self.gen_state_lstm2(gen_lstm2_input, (gc2, gh2))
			gen_out = gen_out.squeeze(0)
			mdn_out = self.gen_state_fc2(gen_out)

			term_out = self.term_fc1(gen_out)
			term_out = self.term_relu1(term_out)
			term_out = self.term_fc2(term_out)
			term_out = self.term_relu2(term_out)
			term_out = self.term_fc3(term_out)
			term = self.term_sigmoid(term_out)

			eos = self.mdn_sigmoid(mdn_out[:, 0])
			[mu1, mu2, sig1, sig2, rho, pi] = torch.split(mdn_out[:, 1:], self.num_mixtures, 1)
			sig1 = sig1.exp() + 1e-3
			sig2 = sig2.exp() + 1e-3
			rho = self.mdn_tanh(rho)
			pi = self.mdn_softmax(pi)
			mus = torch.stack([mu1, mu2], -1).squeeze()

			pi = pi.cpu().detach().numpy()
			mus = mus.cpu().detach().numpy()
			rho = rho.cpu().detach().numpy()[0]
			eos = eos.cpu().detach().numpy()[0]
			term = term.cpu().detach().numpy()[0][0]

			terms.append(term)
			[dx, dy] = np.sum(pi.reshape(20, 1) * mus, 0)
			# print (eos)
			touch = 1 if eos > 0.5 else 0

			commands.append([dx, dy, touch])
			gen_input = torch.FloatTensor([dx, dy, touch]).view([1, 1, 3]).to(self.device)
			character_nums += 1

			# print (zz, term)
			if term > 0.3:
				if target_sentence[current_char_id_count] == ' ':
					current_char_id_count += 1
					character_nums = 0
					if current_char_id_count == len(W_c_rec):
						break
				elif character_nums > 5:
					current_char_id_count += 1
					character_nums = 0
					if current_char_id_count == len(W_c_rec):
						break

			cx += dx * 2.0 * 5.0
			cy += dy * 2.0 * 5.0
			if cx > 1000 or cx < 0:
				break
			if cy > 350 or cy < 0:
				break

		return commands


	def sample_from_w_fix(self, W_c_rec, target_sentence):
		gen_input = torch.zeros([1, 1, 3]).to(self.device)
		current_char_id_count = 0

		gc1 = torch.zeros([self.num_layers, 1, self.weight_dim]).to(self.device)
		gh1 = torch.zeros([self.num_layers, 1, self.weight_dim]).to(self.device)
		gc2 = torch.zeros([self.num_layers, 1, self.weight_dim * 2]).to(self.device)
		gh2 = torch.zeros([self.num_layers, 1, self.weight_dim * 2]).to(self.device)

		terms = []
		commands = []
		character_nums = 0
		cx, cy = 100, 150
		new_char = False
		renewal = False
		for zz in range(800):
			# print (torch.sum(gc1))
			W_c_t_now = W_c_rec[current_char_id_count:current_char_id_count + 1]

			gen_state = self.gen_state_fc1(gen_input)
			gen_state = self.gen_state_relu(gen_state)
			gen_state, (gc1, gh1) = self.gen_state_lstm1(gen_state, (gc1, gh1))
			gen_encoded = gen_state.squeeze(0)

			gen_lstm2_input = torch.cat([gen_encoded, W_c_t_now], -1)
			gen_lstm2_input = gen_lstm2_input.view([1, 1, self.weight_dim * 2])
			gen_out, (gc2, gh2) = self.gen_state_lstm2(gen_lstm2_input, (gc2, gh2))
			gen_out = gen_out.squeeze(0)
			mdn_out = self.gen_state_fc2(gen_out)

			term_out = self.term_fc1(gen_out)
			term_out = self.term_relu1(term_out)
			term_out = self.term_fc2(term_out)
			term_out = self.term_relu2(term_out)
			term_out = self.term_fc3(term_out)
			term = self.term_sigmoid(term_out)

			eos = self.mdn_sigmoid(mdn_out[:, 0])
			[mu1, mu2, sig1, sig2, rho, pi] = torch.split(mdn_out[:, 1:], self.num_mixtures, 1)
			sig1 = sig1.exp() + 1e-3
			sig2 = sig2.exp() + 1e-3
			rho = self.mdn_tanh(rho)
			pi = self.mdn_softmax(pi)
			mus = torch.stack([mu1+self.scale_sd*sig1, mu2+self.scale_sd*sig2], -1).squeeze()

			pi = pi.cpu().detach().numpy()
			mus = mus.cpu().detach().numpy()
			rho = rho.cpu().detach().numpy()[0]
			eos = eos.cpu().detach().numpy()[0]
			term = term.cpu().detach().numpy()[0][0]

			terms.append(term)
			[dx, dy] = np.sum(pi.reshape(20, 1) * mus, 0)
			touch = 1 if eos > 0.5 else 0

			if new_char and touch == 1:
				new_char = False
				commands.append([dx, dy, touch])
				return commands, current_char_id_count
			else:
				commands.append([dx, dy, touch])
				gen_input = torch.FloatTensor([dx, dy, touch]).view([1, 1, 3]).to(self.device)

			character_nums += 1

			# print (zz, term)
			if term > 0.5:
				if character_nums > 5:
					current_char_id_count += 1
					character_nums = 0
					new_char = True
					if current_char_id_count == len(W_c_rec):
						break

			cx += dx * 2.0 * 5.0
			cy += dy * 2.0 * 5.0
			if cx > 1000 or cx < 0:
				break
			if cy > 350 or cy < 0:
				break

		return commands, -1