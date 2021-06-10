import sys
import socket
import os
import shutil
from lib.path import n_thyme_result_path
from lib.input_process import load_econtextsep, load_pickle, get_tokenizer, get_sep2coherent, get_loc, get_label, get_embedding_matrix
from keras.models import Model
from keras.layers import Input, Embedding, Bidirectional, LSTM, Lambda, Dense, Dot, Multiply, Activation, Flatten, Concatenate, Reshape, Dropout
from lib.routing import Routing
import keras.backend as K
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam
from lib.procedure_process import employ_earlystop
from lib.evaluation import get_evaluation
from lib.output_process import save_file, save_result, select_model, save_procedure_parameter, rename_result_folder


def n_c_l_lstm_w_shs_wrourting_R():

	coin_mark = 0  # [0, 1]
	initial_iteration = 10
	initial_coin_bank = 20
	considered_last = 5
	considered_top = 1
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	KTF.set_session(sess)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
	corpus_parameter_list = []
	embedding_type_list = ['full', 'section']
	embedding_dim_list = [100, 200]
	win_scale_list = [2, 4, 6, 8, 10]
	context_scale_list = [2, 4, 6, 8, 10]
	for embedding_type in embedding_type_list:
		for embedding_dim in embedding_dim_list:
			for win_scale in win_scale_list:
				for context_scale in context_scale_list:
					corpus_parameter_list.append({'embedding_type': embedding_type, 'embedding_dim': str(embedding_dim), 'win_scale': str(win_scale), 'context_scale': str(context_scale)})
	network_parameter_list = []
	batch_size_list = [32, 64, 128, 256, 512, 1024, 2048]
	lr_list = [0.001, 0.002]
	optimizer_list = ['Adam']
	in_dropout_list = [0, 0.5]
	out_dropout_list = [0, 0.5] 
	lstm_dim_list = [128, 256, 512] 
	slide_list_list = [[10, 5], [10, 10], [20, 5], [20, 10], [20, 20]] 
	routing_dim_list = [128, 256, 512] 
	routing_epoch_list = [1, 3, 5, 7, 9] 
	for batch_size in batch_size_list:
		for lr in lr_list:
			for optimizer in optimizer_list:
				for in_dropout in in_dropout_list:
					for out_dropout in out_dropout_list:
						for lstm_dim in lstm_dim_list:
							for slide_list in slide_list_list:
								for routing_dim in routing_dim_list:
									for routing_epoch in routing_epoch_list:
										network_parameter_list.append({'batch_size': batch_size, 'lr': lr, 'optimizer': optimizer, 'in_dropout': in_dropout, 'out_dropout': out_dropout, 'lstm_dim': lstm_dim, 'slide_list': slide_list, 'routing_dim': routing_dim, 'routing_epoch': routing_epoch})
	for corpus_parameter in corpus_parameter_list:
		for network_parameter in network_parameter_list:
			n_feature = sys._getframe().f_code.co_name.strip().split('_')[1] + '(' + sys._getframe().f_code.co_name[3 + len(sys._getframe().f_code.co_name.strip().split('_')[1]):].replace('_', '-') + ')'
			n_feature = n_feature.replace('-R', '')
			n_feature += '_s40' + 'cs' + str(corpus_parameter['context_scale']) + 'ws' + str(corpus_parameter['win_scale']) + '_ld' + str(network_parameter['lstm_dim']) + 'sw' + str(network_parameter['slide_list'][0]) + 'ss' + str(network_parameter['slide_list'][1]) + 'sh±1.0' + 'rd' + str(network_parameter['routing_dim']) + 're' + str(network_parameter['routing_epoch']) + '_id' + str(network_parameter['in_dropout']) + 'od' + str(network_parameter['out_dropout'])
			n_feature += '_' + socket.gethostname().strip().split('-')[1]
			n_feature += '_R'
			econtextsep_list = []
			sen_list = []
			win_list = []
			loc_list = []
			ans_list = []
			for data_set in ['train', 'dev', 'test']:
				econtextsep_list.append(load_econtextsep(data_set, corpus_parameter['context_scale']))
				sen_list.append(load_pickle('senf_s40_' + data_set + '_r10-10.pkl'))
				win_list.append(load_pickle('win_w' + corpus_parameter['win_scale'] + '_' + data_set + '_r10-10.pkl'))
				loc_list.append(load_pickle('loc_' + data_set + '_r10-10.pkl'))
				ans_list.append(load_pickle('label_' + data_set + '_r10-10.pkl'))
			vec_feature = corpus_parameter['embedding_type'] + '_d' + corpus_parameter['embedding_dim'] + 'i5w5'
			vec_dic = load_pickle('vec_' + vec_feature + '.pkl')
			MAX_NB_WORDS = 20000
			MAX_SEQUENCE_LENGTH = 40
			MAX_WIN_LENGTH = int(corpus_parameter['win_scale']) * 2 + 3
			texts = []
			for i in range(3):
				for j in range(len(econtextsep_list[i])):
					texts += econtextsep_list[i][j]
			word_index, data_train_list = get_tokenizer(MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, texts, econtextsep_list[0])
			data_dev_list = get_tokenizer(MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, texts, econtextsep_list[1])[1]
			data_test_list = get_tokenizer(MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, texts, econtextsep_list[2])[1]
			data_train_list = get_sep2coherent(data_train_list, set_len=1)
			data_dev_list = get_sep2coherent(data_dev_list, set_len=1)
			data_test_list = get_sep2coherent(data_test_list, set_len=1)
			data_sen_list = get_tokenizer(MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, texts, sen_list)[1]
			data_win_list = get_tokenizer(MAX_NB_WORDS, MAX_WIN_LENGTH, texts, win_list)[1]
			LOC_LENGTH, loc_train_list = get_loc((2 * int(corpus_parameter['context_scale']) + 1) * MAX_SEQUENCE_LENGTH, loc_list[0])
			loc_dev_list = get_loc((2 * int(corpus_parameter['context_scale']) + 1) * MAX_SEQUENCE_LENGTH, loc_list[1])[1]
			loc_test_list = get_loc((2 * int(corpus_parameter['context_scale']) + 1) * MAX_SEQUENCE_LENGTH, loc_list[2])[1]
			loc_sen_train_list = get_loc(MAX_SEQUENCE_LENGTH, loc_list[0])[1]
			loc_sen_dev_list = get_loc(MAX_SEQUENCE_LENGTH, loc_list[1])[1]
			loc_sen_test_list = get_loc(MAX_SEQUENCE_LENGTH, loc_list[2])[1]
			loc_win_train_list = get_loc(MAX_WIN_LENGTH, loc_list[0])[1]
			loc_win_dev_list = get_loc(MAX_WIN_LENGTH, loc_list[1])[1]
			loc_win_test_list = get_loc(MAX_WIN_LENGTH, loc_list[2])[1]
			y_train = get_label(ans_list[0])
			num_words = min(MAX_NB_WORDS, len(word_index))
			embedding_matrix = get_embedding_matrix(vec_dic, word_index, num_words, int(corpus_parameter['embedding_dim']))
			def get_slice(input_tensor, index):
				output_tensor = input_tensor[:, (index * network_parameter['slide_list'][1]):(index * network_parameter['slide_list'][1] + network_parameter['slide_list'][0]), :]
				return output_tensor
			def get_sim_score(input_tensors):
				dot_tensor = K.batch_dot(input_tensors[0], input_tensors[1], axes=1)
				output_tensor = K.hard_sigmoid(dot_tensor)
				return output_tensor
			def set_sim_score_repeat(input_tensor):
				output_tensor = K.repeat_elements(input_tensor, 2 * network_parameter['lstm_dim'], 2)
				return output_tensor
			def set_sim_score_2dto3d(input_tensor):
				output_tensor = K.reshape(input_tensor, [-1, int(((int(corpus_parameter['context_scale']) * 2 + 1) * MAX_SEQUENCE_LENGTH - network_parameter['slide_list'][0]) / network_parameter['slide_list'][1] + 1), 1])
				return output_tensor
			input_context_layer = Input(shape=((2 * int(corpus_parameter['context_scale']) + 1) * MAX_SEQUENCE_LENGTH,), dtype='float32', name='input_context_layer')
			input_sen_layer = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32', name='input_sen_layer')
			input_win_layer = Input(shape=(MAX_WIN_LENGTH,), dtype='float32', name='input_win_layer')
			input_loc_layer = Input(shape=((2 * int(corpus_parameter['context_scale']) + 1) * MAX_SEQUENCE_LENGTH, LOC_LENGTH), dtype='float32', name='input_loc_layer')
			input_loc_sen_layer = Input(shape=(MAX_SEQUENCE_LENGTH, LOC_LENGTH), dtype='float32', name='input_loc_sen_layer')
			input_loc_win_layer = Input(shape=(MAX_WIN_LENGTH, LOC_LENGTH), dtype='float32', name='input_loc_win_layer')
			shared_embedding = Embedding(input_dim=num_words, output_dim=int(corpus_parameter['embedding_dim']), weights=[embedding_matrix], mask_zero=False, trainable=True)
			embedding_context_layer = shared_embedding(input_context_layer)
			embedding_sen_layer = shared_embedding(input_sen_layer)
			embedding_win_layer = shared_embedding(input_win_layer)
			loced_embedding_context_layer = Concatenate(axis=-1)([embedding_context_layer, input_loc_layer])
			loced_embedding_sen_layer = Concatenate(axis=-1)([embedding_sen_layer, input_loc_sen_layer])
			loced_embedding_win_layer = Concatenate(axis=-1)([embedding_win_layer, input_loc_win_layer])
			if network_parameter['in_dropout'] != 0:
				loced_embedding_context_layer = Dropout(rate=network_parameter['in_dropout'])(loced_embedding_context_layer)
			lstm_sen_layer = Bidirectional(LSTM(network_parameter['lstm_dim']))(loced_embedding_sen_layer)
			lstm_win_layer = Bidirectional(LSTM(network_parameter['lstm_dim']))(loced_embedding_win_layer)
			lstm_layer = np.array([])
			shared_lstm = Bidirectional(LSTM(network_parameter['lstm_dim']))
			for i in range(((2 * int(corpus_parameter['context_scale']) + 1) * 40 - network_parameter['slide_list'][0]) // network_parameter['slide_list'][1] + 1):
				slice_layer = Lambda(get_slice, arguments={'index': i})(loced_embedding_context_layer)
				slice_lstm_layer = shared_lstm(slice_layer)
				slice_lstm_3d_layer = Reshape((1, 2 * network_parameter['lstm_dim']))(slice_lstm_layer)
				sim_score_layer = Lambda(get_sim_score)([lstm_sen_layer, slice_lstm_layer])
				sim_score_3d_layer = Reshape((1, 1))(sim_score_layer)
				sim_score_3d_repeat_layer = Lambda(set_sim_score_repeat)(sim_score_3d_layer)
				scored_slice_lstm_layer = Multiply()([slice_lstm_3d_layer, sim_score_3d_repeat_layer])
				if i == 0:
					lstm_layer = scored_slice_lstm_layer
				else:
					lstm_layer = Concatenate(axis=1)([lstm_layer, scored_slice_lstm_layer])
			sim_score_layer = Dot(axes=[2, 1])([lstm_layer, lstm_win_layer])
			sim_score_activation_layer = Activation('sigmoid')(sim_score_layer)
			sim_score_activation_3d_layer = Lambda(set_sim_score_2dto3d)(sim_score_activation_layer)
			sim_score_activation_3d_rep_layer = Lambda(set_sim_score_repeat)(sim_score_activation_3d_layer)
			score_lstm_layer = Multiply()([lstm_layer, sim_score_activation_3d_rep_layer])
			routing_layer = Routing(routing_dim=network_parameter['routing_dim'], routing_epoch=network_parameter['routing_epoch'])(score_lstm_layer)
			output_layer = Flatten()(routing_layer)
			if network_parameter['out_dropout'] != 0:
				output_layer = Dropout(rate=network_parameter['out_dropout'])(output_layer)
			prediction_layer = Dense(4, activation='softmax')(output_layer)
			model = Model(inputs=[input_context_layer, input_sen_layer, input_win_layer, input_loc_layer, input_loc_sen_layer, input_loc_win_layer], outputs=prediction_layer)
			if network_parameter['optimizer'] == 'Adam':
				model.compile(optimizer=Adam(lr=network_parameter['lr']), loss=['categorical_crossentropy'], metrics=['accuracy'])
			print()
			model.summary()
			print('corpus_parameter:' + str(corpus_parameter))
			print('network_parameter:' + str(network_parameter))
			if os.path.exists(n_thyme_result_path + '_' + n_feature):
				shutil.rmtree(n_thyme_result_path + '_' + n_feature)
			os.makedirs(str(n_thyme_result_path + '_' + n_feature))
			os.makedirs(str(n_thyme_result_path + '_' + n_feature) + '/codebackups/')
			os.makedirs(str(n_thyme_result_path + '_' + n_feature) + '/models/')
			os.makedirs(str(n_thyme_result_path + '_' + n_feature) + '/parameters/')
			os.makedirs(str(n_thyme_result_path + '_' + n_feature) + '/results/')
			save_file(n_feature, sys._getframe().f_code.co_name.strip().split('_')[1], sys._getframe().f_code.co_name[2:])
			f_parameter = open(str(n_thyme_result_path + '_' + n_feature) + '/parameters/' + 'parameter.txt', 'w', encoding='UTF-8')
			f_parameter.write('corpus_parameter:\n')
			f_parameter.write(str(corpus_parameter) + '\n\n')
			f_parameter.write('network_parameter:\n')
			f_parameter.write(str(network_parameter) + '\n\n')
			f_parameter.close()
			f_result_backup_dev = open(str(n_thyme_result_path + '_' + n_feature) + '/results/' + 'results_backup_dev.txt', 'w', encoding='UTF-8')
			f_result_backup_test = open(str(n_thyme_result_path + '_' + n_feature) + '/results/' + 'results_backup_test.txt', 'w', encoding='UTF-8')
			f_result_backup_dev.write('dct_dev_list' + '\t' + 'before_dev_list' + '\t' + 'after_dev_list' + '\t' + 'overlap_dev_list' + '\t' + 'beforeoverlap_dev_list' + '\n')
			f_result_backup_test.write('dct_test_list' + '\t' + 'before_test_list' + '\t' + 'after_test_list' + '\t' + 'overlap_test_list' + '\t' + 'beforeoverlap_test_list' + '\n')
			e = 0
			iteration = initial_iteration
			coin_bank = initial_coin_bank
			if coin_mark == 0:
				coin_bank = 0
			dev_result_list = []
			test_result_list = []
			dev_result_dic = {}
			test_result_dic = {}
			while e != iteration:
				e += 1
				model.fit({'input_context_layer': data_train_list, 'input_sen_layer': data_sen_list[0], 'input_win_layer': data_win_list[0], 'input_loc_layer': loc_train_list, 'input_loc_sen_layer': loc_sen_train_list, 'input_loc_win_layer': loc_win_train_list}, [y_train], batch_size=network_parameter['batch_size'], verbose=1)
				prediction_dev_list = model.predict({'input_context_layer': data_dev_list, 'input_sen_layer': data_sen_list[1], 'input_win_layer': data_win_list[1], 'input_loc_layer': loc_dev_list, 'input_loc_sen_layer': loc_sen_dev_list, 'input_loc_win_layer': loc_win_dev_list}, batch_size=1024).tolist()
				prediction_test_list = model.predict({'input_context_layer': data_test_list, 'input_sen_layer': data_sen_list[2], 'input_win_layer': data_win_list[2], 'input_loc_layer': loc_test_list, 'input_loc_sen_layer': loc_sen_test_list, 'input_loc_win_layer': loc_win_test_list}, batch_size=1024).tolist()
				predicted_label_dev_list = []
				for p in range(len(prediction_dev_list)):
					predicted_label_dev_list.append(prediction_dev_list[p].index(max(prediction_dev_list[p])))
				predicted_label_test_list = []
				for p in range(len(prediction_test_list)):
					predicted_label_test_list.append(prediction_test_list[p].index(max(prediction_test_list[p])))
				dct_dev_list, before_dev_list, after_dev_list, overlap_dev_list, beforeoverlap_dev_list = get_evaluation(predicted_label_dev_list, ans_list[1])
				dct_test_list, before_test_list, after_test_list, overlap_test_list, beforeoverlap_test_list = get_evaluation(predicted_label_test_list, ans_list[2])
				dev_result_list.append(dct_dev_list[0])
				test_result_list.append(dct_test_list[0])
				dev_result_dic[e] = dct_dev_list[0]
				test_result_dic[e] = dct_test_list[0]
				model.save(str(n_thyme_result_path + '_' + n_feature) + '/models/saved_model_' + str(e) + '.h5')
				save_result(n_feature, prediction_test_list, predicted_label_test_list, dct_dev_list, dct_test_list, before_dev_list, after_dev_list, overlap_dev_list, beforeoverlap_dev_list, before_test_list, after_test_list, overlap_test_list, beforeoverlap_test_list, e)
				f_result_backup_dev.write(str(e) + '\t' + str(dct_dev_list) + '\t' + str(before_dev_list) + '\t' + str(after_dev_list) + '\t' + str(overlap_dev_list) + '\t' + str(beforeoverlap_dev_list) + '\n')
				f_result_backup_test.write(str(e) + '\t' + str(dct_test_list) + '\t' + str(before_test_list) + '\t' + str(after_test_list) + '\t' + str(overlap_test_list) + '\t' + str(beforeoverlap_test_list) + '\n')
				coin_bank, iteration = employ_earlystop(dev_result_list, coin_bank, e, iteration, considered_top, considered_last)
			max_e_dev, true_max_e_test, max_dct_dev_str, max_dct_test_str, true_max_dct_test_str = select_model(n_feature, dev_result_dic, test_result_dic)
			save_procedure_parameter(f_result_backup_dev, f_result_backup_test, coin_mark, initial_iteration, initial_coin_bank, considered_top, considered_last, max_e_dev, true_max_e_test, dev_result_dic)
			f_result_backup_dev.close()
			f_result_backup_test.close()
			rename_result_folder(n_feature, max_dct_dev_str, max_dct_test_str, true_max_dct_test_str)


