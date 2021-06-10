import os
import pickle
from lib.path import n_thyme_data_path
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import np_utils


def load_pickle(file):
	f = open(os.path.join(n_thyme_data_path, file), 'rb')
	loaded_data = pickle.load(f)
	f.close()
	return loaded_data


def load_econtextsep(data_set, context_scale):
	the_file_list = []
	for i in range(int(context_scale)):
		the_list = []
		current_list = load_pickle('contextleft_l' + str(i + 1) + '_' + data_set + '_r10-10.pkl')
		for j in range(len(current_list)):
			the_text = ''
			word_list = current_list[j].strip().split(' ')
			if len(word_list) > 40:
				for word in word_list[0:40]:
					the_text += word + ' '
				the_list.append(the_text.strip())
			else:
				the_list.append(current_list[j])
		the_file_list.append(the_list)
	the_file_list.append(load_pickle('senf_s40_' + data_set + '_r10-10.pkl'))
	for i in range(int(context_scale)):
		the_list = []
		current_list = load_pickle('contextright_l' + str(i + 1) + '_' + data_set + '_r10-10.pkl')
		for j in range(len(current_list)):
			the_text = ''
			word_list = current_list[j].strip().split(' ')
			if len(word_list) > 40:
				for word in word_list[0:40]:
					the_text += word + ' '
				the_list.append(the_text.strip())
			else:
				the_list.append(current_list[j])
		the_file_list.append(the_list)
	return the_file_list


def get_tokenizer(MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, texts, text_list):
	tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='\n')
	tokenizer.fit_on_texts(texts)
	word_index = tokenizer.word_index
	data_list = []
	for i in range(len(text_list)):
		sequence = tokenizer.texts_to_sequences(text_list[i])
		data = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
		data_list.append(data)
	return word_index, np.array(data_list)


def get_sep2coherent(data_list, set_len):
	sep_array = data_list
	coherent_array = np.array([])
	if set_len == 1:
		for i in range(len(sep_array)):
			if i == 0:
				coherent_array = sep_array[i]
			else:
				coherent_array = np.hstack((coherent_array, sep_array[i]))
	elif set_len == 3:
		pass
	return coherent_array


def get_loc(MAX_SEQUENCE_LENGTH, loc_list):
	loc_onehot = []
	for i in range(len(loc_list)):
		loc_list = [0] * 15
		loc_list[loc_list[i]] = 1
		repeat_loc_list = []
		for j in range(MAX_SEQUENCE_LENGTH):
			repeat_loc_list.append(loc_list)
		loc_onehot.append(repeat_loc_list)
	loc_data = np.array(loc_onehot)
	LOC_LENGTH = loc_data.shape[-1]
	return LOC_LENGTH, loc_data


def get_label(ans):
	lable = np_utils.to_categorical(np.asarray(ans))
	return lable


def get_embedding_matrix(vec_dic, word_index, num_words, embedding_dim):
	embedding_matrix = np.zeros((num_words, embedding_dim))
	for word, i in word_index.items():
		if i >= num_words:
			continue
		embedding_vector = vec_dic.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	return embedding_matrix


