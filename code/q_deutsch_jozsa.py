## Deutsch-Jozsa Algorithm
# - Simon Gruening

# python v3.*
import numpy as np
import itertools
import random

# np.set_printoptions(suppress=True)


## Required gates
pauli_x_gate = [ [0,1],[1,0] ]
hadamard_gate = (1.0/(2**0.5)) * np.array([[1,1],[1,-1]])


## Prepare States
s0 = np.array( [[1.0],[0.0]] )
s1 = np.matmul(pauli_x_gate,s0)
# s1 = np.array( [[0.0],[1.0]] )


## Function f

# number of _bits_ f takes as input
n = 2
print("n =",n)

# function to be tested
# ex. for n=2:  f = {(0,0):0,(0,1):0,(1,0):0,(1,1):0} -> P=1

f_index = list(itertools.product([0, 1], repeat=n))
f = {k:0 for k in f_index}

if random.random()>0.5:
 	# make balanced
	ones_i = np.random.choice(len(f_index),size=int(len(f_index)/2),replace=False)
	ones = [f_index[i] for i in ones_i]
	for o in ones:
		f[o] = 1

print("f:",f)


## Auxiliary functions

# tensor a list of arrays in given order
def kron(states):
	result = np.array([[1.0]])
	for s in states:
		result = np.kron(result,s)
	return result

# translate binary tuple to qubit array
def binary_to_qubit(binary_tuple): 
	return kron([s0 if x == 0 else s1 for x in binary_tuple])


## Prepare state |00...1> = |x>|y> with y as the final qubit
prepared_state = kron((n*[s0])+[s1])
print("Psi_0: \n",prepared_state)

# apply hadamard individually to each qubit
tensored_h = kron([hadamard_gate]*(n+1))
print("Hadamard^(x)(n+1): \n",tensored_h)

pre_oracle_state = np.matmul(tensored_h,prepared_state)
print("Psi_1: \n",pre_oracle_state)

# print("norm:",np.linalg.norm(prepared_state))
# print("norm:",np.linalg.norm(pre_oracle_state))
# very slight loss due to accuracy of floats =)


## Create Oracle

# x,y -> x,y (+) f(x) where y is the last qubit
# 	(+) is addition modulo 2
# column-wise creation
lst = list(itertools.product([0, 1], repeat=n+1)) # note order

oracle = np.array([],dtype=float)
for tmp in lst:
	x = tmp[0:-1]
	y = tmp[-1]
	f_x = f[x]

	qx = binary_to_qubit(x)
	qf_x = binary_to_qubit((f_x,))

	if y == 1:
		# addition modulo 2
		qf_x = np.mod(qf_x+1,2)

	column = kron([qx,qf_x])
	
	column = np.array([j[0] for j in column])

	oracle = np.concatenate((oracle,column)) #,axis=0)

qn = 2**(n+1)
oracle = np.transpose(oracle.reshape((qn,qn)))
print("Oracle: \n",oracle)


## Apply last two gates

# apply oracle
post_oracle_state = np.matmul(oracle,pre_oracle_state)
print("Psi_2: \n",post_oracle_state)

# apply hadamard to all but last (y) qubit
tensored_h_id = kron([hadamard_gate]*(n)+[np.identity(2)])
print("Hadamard^(x)n (x) id_2: \n", tensored_h_id)
end_state = np.matmul(tensored_h_id,post_oracle_state)

print("Psi_3: \n",end_state)


## Measure 
# "ignore" last qubit (y)

print("Numerical error on norm: ",1-np.linalg.norm(end_state))

# projectors for |00..00y>, i.e. <y00...00| with y in {0,1}
P000 = np.transpose(kron([s0]*n+[s0]))
P001 = np.transpose(kron([s0]*n+[s1]))

# apply projectors
prob_000 = np.dot(end_state.reshape(2**(n+1)),P000.reshape(2**(n+1)))**2
prob_001 = np.dot(end_state.reshape(2**(n+1)),P001.reshape(2**(n+1)))**2
prob_total = prob_000+prob_001

print("Probability of |00...00>|0>: ",prob_000)
print("Probability of |00...00>|1>: ",prob_001)
print("Probability of |00...00>|y>: ",prob_total)

if round(prob_total) == 1:
	print("Thus the function is constant.")
else:
	print("Thus the function is balanced.")
