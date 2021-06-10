def get_evaluation(predicted_label_list, ins_ans):
	dct_up = 0
	dct_down = 0
	before_p_up = 0
	before_p_down = 0
	before_r_up = 0
	before_r_down = 0
	after_p_up = 0
	after_p_down = 0
	after_r_up = 0
	after_r_down = 0
	overlap_p_up = 0
	overlap_p_down = 0
	overlap_r_up = 0
	overlap_r_down = 0
	beforeoverlap_p_up = 0
	beforeoverlap_p_down = 0
	beforeoverlap_r_up = 0
	beforeoverlap_r_down = 0
	for i in range(len(ins_ans)):
		if ins_ans[i][0] == predicted_label_list[i]:
			dct_up += 1
		if ins_ans[i][0] in [0, 1, 2, 3]:
			dct_down += 1
		if (ins_ans[i][0] == 0) and (predicted_label_list[i] == 0):
			before_p_up += 1
			before_r_up += 1
		if predicted_label_list[i] == 0:
			before_p_down += 1
		if ins_ans[i][0] == 0:
			before_r_down += 1
		if (ins_ans[i][0] == 1) and (predicted_label_list[i] == 1):
			after_p_up += 1
			after_r_up += 1
		if predicted_label_list[i] == 1:
			after_p_down += 1
		if ins_ans[i][0] == 1:
			after_r_down += 1
		if (ins_ans[i][0] == 2) and (predicted_label_list[i] == 2):
			overlap_p_up += 1
			overlap_r_up += 1
		if predicted_label_list[i] == 2:
			overlap_p_down += 1
		if ins_ans[i][0] == 2:
			overlap_r_down += 1
		if (ins_ans[i][0] == 3) and (predicted_label_list[i] == 3):
			beforeoverlap_p_up += 1
			beforeoverlap_r_up += 1
		if predicted_label_list[i] == 3:
			beforeoverlap_p_down += 1
		if ins_ans[i][0] == 3:
			beforeoverlap_r_down += 1
	if dct_down != 0:
		dct = round((dct_up / dct_down), 3)
	else:
		dct = 0
	if before_p_down != 0:
		before_p = round((before_p_up / before_p_down), 3)
	else:
		before_p = 0
	if before_r_down != 0:
		before_r = round((before_r_up / before_r_down), 3)
	else:
		before_r = 0
	if (before_p != 0) or (before_r != 0):
		before_f = round((2 * (before_p_up / before_p_down) * (before_r_up / before_r_down)) / ((before_p_up / before_p_down) + (before_r_up / before_r_down)), 3)
	else:
		before_f = 0
	if after_p_down != 0:
		after_p = round((after_p_up / after_p_down), 3)
	else:
		after_p = 0
	if after_r_down != 0:
		after_r = round((after_r_up / after_r_down), 3)
	else:
		after_r = 0
	if (after_p != 0) or (after_r != 0):
		after_f = round((2 * (after_p_up / after_p_down) * (after_r_up / after_r_down)) / ((after_p_up / after_p_down) + (after_r_up / after_r_down)), 3)
	else:
		after_f = 0
	if overlap_p_down != 0:
		overlap_p = round((overlap_p_up / overlap_p_down), 3)
	else:
		overlap_p = 0
	if overlap_r_down != 0:
		overlap_r = round((overlap_r_up / overlap_r_down), 3)
	else:
		overlap_r = 0
	if (overlap_p != 0) or (overlap_r != 0):
		overlap_f = round((2 * (overlap_p_up / overlap_p_down) * (overlap_r_up / overlap_r_down)) / ((overlap_p_up / overlap_p_down) + (overlap_r_up / overlap_r_down)), 3)
	else:
		overlap_f = 0
	if beforeoverlap_p_down != 0:
		beforeoverlap_p = round((beforeoverlap_p_up / beforeoverlap_p_down), 3)
	else:
		beforeoverlap_p = 0
	if beforeoverlap_r_down != 0:
		beforeoverlap_r = round((beforeoverlap_r_up / beforeoverlap_r_down), 3)
	else:
		beforeoverlap_r = 0
	if (beforeoverlap_p != 0) or (beforeoverlap_r != 0):
		beforeoverlap_f = round((2 * (beforeoverlap_p_up / beforeoverlap_p_down) * (beforeoverlap_r_up / beforeoverlap_r_down)) / ((beforeoverlap_p_up / beforeoverlap_p_down) + (beforeoverlap_r_up / beforeoverlap_r_down)), 3)
	else:
		beforeoverlap_f = 0
	return [dct, dct_up, dct_down, dct_down], [before_f, before_p, before_r, before_p_up, before_p_down, before_r_down], [after_f, after_p, after_r, after_p_up, after_p_down, after_r_down], [overlap_f, overlap_p, overlap_r, overlap_p_up, overlap_p_down, overlap_r_down], [beforeoverlap_f, beforeoverlap_p, beforeoverlap_r, beforeoverlap_p_up, beforeoverlap_p_down, beforeoverlap_r_down]


