# -*- coding: utf-8 -*-
import numpy as np

def parallel(*args):
	"""
	Inputs:
		*args: Unpacked tuple of resistances.
	Returns:
		Float. Parallel combination of all of the arguments in *args.
	"""
	try:
		return 1/sum([1/a for a in args])
	except:
		return 0

def zero_crossing(lst):
	return np.where(np.diff(np.sign(lst)))[0][0]

def cond_print(myStr, yesPrint=True):
	"""
	Inputs:
		myStr: String.
		yesPrint: Boolean.
	Returns:
		None.
	"""
	if yesPrint:
		print(myStr)
	return
