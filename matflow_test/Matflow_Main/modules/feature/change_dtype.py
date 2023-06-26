import pandas as pd
from django.http import JsonResponse

from ...modules import utils
from ...modules.classes import dtype_changer

def change_dtype(file):
	print (file)
	data = file.get("data")
	data = pd.DataFrame(data)
# 	variables = utils.get_variables(data)
# 	orig_dtypes = utils.get_dtypes(data)
# 	var_with_dtype = {f"{var} ({dtype})": var for (var, dtype) in zip(variables, orig_dtypes)}
# 	cat_var = utils.get_categorical(data)
#
# 	n_iter = file.get("number_of_columns")
# 	change_dict = {}
# 	temp_array=file.get("values")
# 	for i in range(int(n_iter)):
# 		var =temp_array[i].get("column")
# 		desired_dtype = temp_array[i].get("desired_dtype")
# 		if desired_dtype in ["int", "float"]:
# 			desired_bits = temp_array.get("desired_dtype")
# 		else:
# 			desired_bits =temp_array[i].get("desired_dtype")
# 		change_dict[var_with_dtype[var]] = desired_dtype + desired_bits
# 		selected = list(change_dict.keys())
# 		var_with_dtype = {key: val for (key, val) in var_with_dtype.items() if val not in selected}
#
# 	status = [change_check(data, var, dtype) for (var, dtype) in change_dict.items()]
#
# 	if all(status):
# 		chg = dtype_changer.DtypeChanger(change_dict)
# 		new_value = chg.fit_transform(data)
	df = data.to_dict(orient="records")
	return JsonResponse(df, safe=False)
#
# def change_check(data, var, dtype):
# 	try:
# 		data[var].astype(dtype)
# 		return True
# 	except:
# 		return False
