def employ_earlystop(dev_result_list, coin_bank, e, iteration, considered_top, considered_last):
	if (e == iteration) and (coin_bank != 0):
		last_list = dev_result_list[-considered_last:]
		dev_result_list.sort()
		top_list = dev_result_list[-considered_top:]
		for top in top_list:
			if top in last_list:
				iteration += 5
				break
		coin_bank -= 1
	return coin_bank, iteration


