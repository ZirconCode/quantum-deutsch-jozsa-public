
# python v3.*
import numpy as np
import itertools
import random

n=4

## Create Function
f_index = list(itertools.product([0, 1], repeat=n))
f = {k:0 for k in f_index}

if random.random()>0.5:
 	# make balanced
	ones_i = np.random.choice(len(f_index),size=int(len(f_index)/2),replace=False)
	ones = [f_index[i] for i in ones_i]
	for o in ones:
		f[o] = 1

print("f:",f)


## Check if Balanced
count = 0
state = -1
for i in f.keys():
	if state == -1:
		state = f[i]
	elif state == f[i]:
		count = count+1
		if count>(len(f.keys())/2): # maximum iterations
			print("Constant")
			break
	else:
		print("Balanced")
		break

