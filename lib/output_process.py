import os
import shutil
from lib.path import n_root_path, n_thyme_result_path


def save_file(n_feature, scale, network_file):
	if scale == 'c':
		scale = 'context'
	if scale == 's':
		scale = 'sen'
	if scale == 'w':
		scale = 'win'
	if scale == 'sc':
		scale = 'surrounding_context'
	os.makedirs(str(n_thyme_result_path + '_' + n_feature) + '/codebackups/' + scale + '_scale' + '/')
	shutil.copyfile(n_root_path + 'networks/' + scale + '_scale' + '/' + network_file + '.py', str(n_thyme_result_path + '_' + n_feature) + '/codebackups/' + scale + '_scale' + '/' + network_file + '.py')
	shutil.copytree(n_root_path + 'lib/', str(n_thyme_result_path + '_' + n_feature) + '/codebackups/lib/')
	if os.path.exists(str(n_thyme_result_path + '_' + n_feature) + '/codebackups/lib/__pycache__'):
		shutil.rmtree(str(n_thyme_result_path + '_' + n_feature) + '/codebackups/lib/__pycache__')


def save_result(n_feature, prediction_test_list, predicted_label_test_list, dct_dev_list, dct_test_list, before_dev_list, after_dev_list, overlap_dev_list, beforeoverlap_dev_list, before_test_list, after_test_list, overlap_test_list, beforeoverlap_test_list, e):
	dct_dev = str(dct_dev_list[0])
	if len(dct_dev) < 5:
		for i in range(5 - len(dct_dev)):
			dct_dev += '0'
	f_result_dev = open(n_thyme_result_path + '_' + n_feature + '/' + 'dev_' + dct_dev + '_b' + str(before_dev_list[0]) + '_o' + str(overlap_dev_list[0]) + '_bo' + str(beforeoverlap_dev_list[0]) + '_a' + str(after_dev_list[0]) + '_' + str(e) + '.txt', 'w', encoding='UTF-8')
	f_result_dev.write('dct_dev_list' + '\t' + str(dct_dev_list) + '\n')
	f_result_dev.write('before_dev_list' + '\t' + str(before_dev_list) + '\n')
	f_result_dev.write('after_dev_list' + '\t' + str(after_dev_list) + '\n')
	f_result_dev.write('overlap_dev_list' + '\t' + str(overlap_dev_list) + '\n')
	f_result_dev.write('beforeoverlap_dev_list' + '\t' + str(beforeoverlap_dev_list) + '\n')
	for p in range(len(predicted_label_test_list)):
		f_result_dev.write(str(predicted_label_test_list[p]) + '\t' + str(prediction_test_list[p]) + '\n')
	f_result_dev.close()
	dct_test = str(dct_test_list[0])
	if len(dct_test) < 5:
		for i in range(5 - len(dct_test)):
			dct_test += '0'
	f_result_test = open(n_thyme_result_path + '_' + n_feature + '/' + 'test_' + dct_test + '_b' + str(before_test_list[0]) + '_o' + str(overlap_test_list[0]) + '_bo' + str(beforeoverlap_test_list[0]) + '_a' + str(after_test_list[0]) + '_' + str(e) + '.txt', 'w', encoding='UTF-8')
	f_result_test.write('dct_test_list' + '\t' + str(dct_test_list) + '\n')
	f_result_test.write('before_test_list' + '\t' + str(before_test_list) + '\n')
	f_result_test.write('after_test_list' + '\t' + str(after_test_list) + '\n')
	f_result_test.write('overlap_test_list' + '\t' + str(overlap_test_list) + '\n')
	f_result_test.write('beforeoverlap_test_list' + '\t' + str(beforeoverlap_test_list) + '\n')
	for p in range(len(predicted_label_test_list)):
		f_result_test.write(str(predicted_label_test_list[p]) + '\t' + str(prediction_test_list[p]) + '\n')
	f_result_test.close()


def select_model(n_feature, dev_result_dic, test_result_dic):
	max_dct_dev = 0
	max_e_dev = 0
	for k in dev_result_dic:
		if k <= 3:
			continue
		if (dev_result_dic[k] > max_dct_dev) or ((dev_result_dic[k] == max_dct_dev) and (k > max_e_dev)):
			max_dct_dev = dev_result_dic[k]
			max_e_dev = k
	max_dct_dev_str = str(max_dct_dev)
	if len(max_dct_dev_str) < 5:
		for i in range(5 - len(max_dct_dev_str)):
			max_dct_dev_str += '0'
	max_dct_test_str = str(test_result_dic[max_e_dev])
	if len(max_dct_test_str) < 5:
		for i in range(5 - len(max_dct_test_str)):
			max_dct_test_str += '0'
	true_max_dct_test = 0
	true_max_e_test = 0
	for k in test_result_dic:
		if (test_result_dic[k] > true_max_dct_test) or ((test_result_dic[k] == true_max_dct_test) and (k > true_max_e_test)):
			true_max_dct_test = test_result_dic[k]
			true_max_e_test = k
	true_max_dct_test_str = str(true_max_dct_test)
	if len(true_max_dct_test_str) < 5:
		for i in range(5 - len(true_max_dct_test_str)):
			true_max_dct_test_str += '0'
	saved_e_list = [max_e_dev, true_max_e_test]
	for k in dev_result_dic:
		if (dev_result_dic[k] == max_dct_dev) and (k not in saved_e_list):
			saved_e_list.append(k)
	for k in test_result_dic:
		if (test_result_dic[k] == true_max_dct_test) and (k not in saved_e_list):
			saved_e_list.append(k)
	for file in os.listdir(str(n_thyme_result_path + '_' + n_feature) + '/models/'):
		if int(file[0:-3].strip().split('_')[-1]) not in saved_e_list:
			os.remove(str(n_thyme_result_path + '_' + n_feature) + '/models/' + file)
	for file in os.listdir(str(n_thyme_result_path + '_' + n_feature)):
		if (os.path.isfile(str(n_thyme_result_path + '_' + n_feature) + '/' + file)) and (int(file[0:-4].strip().split('_')[-1]) not in saved_e_list):
			os.remove(str(n_thyme_result_path + '_' + n_feature) + '/' + file)
	return max_e_dev, true_max_e_test, max_dct_dev_str, max_dct_test_str, true_max_dct_test_str


def save_procedure_parameter(f_result_backup_dev, f_result_backup_test, coin_mark, initial_iteration, initial_coin_bank, considered_top, considered_last, max_e_dev, true_max_e_test, dev_result_dic):
	f_result_backup_dev.write('early stop:\n')
	f_result_backup_dev.write(str(coin_mark) + '\n')
	f_result_backup_test.write('early stop:\n')
	f_result_backup_test.write(str(coin_mark) + '\n')
	f_result_backup_dev.write('initial_iteration:\n')
	f_result_backup_dev.write(str(initial_iteration) + '\n')
	f_result_backup_test.write('initial_iteration:\n')
	f_result_backup_test.write(str(initial_iteration) + '\n')
	f_result_backup_dev.write('coin_bank:\n')
	f_result_backup_dev.write(str(initial_coin_bank) + '\n')
	f_result_backup_test.write('coin_bank:\n')
	f_result_backup_test.write(str(initial_coin_bank) + '\n')
	f_result_backup_dev.write('considered_last:\n')
	f_result_backup_dev.write(str(considered_last) + '\n')
	f_result_backup_test.write('considered_last:\n')
	f_result_backup_test.write(str(considered_last) + '\n')
	f_result_backup_dev.write('considered_top:\n')
	f_result_backup_dev.write(str(considered_top) + '\n')
	f_result_backup_test.write('considered_top:\n')
	f_result_backup_test.write(str(considered_top) + '\n')
	f_result_backup_dev.write('epochs:\n')
	f_result_backup_dev.write(str(max_e_dev) + '(' + str(true_max_e_test) + ')-' + str(len(dev_result_dic)) + '\n\n')
	f_result_backup_test.write('epochs:\n')
	f_result_backup_test.write(str(max_e_dev) + '(' + str(true_max_e_test) + ')-' + str(len(dev_result_dic)) + '\n\n')


def rename_result_folder(n_feature, max_dct_dev_str, max_dct_test_str, true_max_dct_test_str):
	old_name_path = n_thyme_result_path + '_' + n_feature
	new_name_path = n_thyme_result_path + max_dct_test_str + '_t' + true_max_dct_test_str + '_d' + max_dct_dev_str + '_' + n_feature
	if os.path.exists(new_name_path):
		folder_serial = 1
		while os.path.exists(new_name_path + '_' + str(folder_serial)):
			folder_serial += 1
		new_name_path += '_' + str(folder_serial)
	os.rename(old_name_path, new_name_path)


