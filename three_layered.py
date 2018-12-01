import numpy as np


def sigmoid(x, deriv=False):
	if(deriv == False):
		return 1/(1+np.exp(-x))
	return x*(1-x)

def main():
	np.random.seed(0)
	h = np.random.random((3, 1))
	w = np.random.random((4, 1))
	#print(h, w)

	X = np.array([[150, 100, 100],
				  [100, 150, 125],
				  [160, 100, 140],
				  [100, 100, 100]])
	y = np.array([[1]])

	for i in range(1):
		l0 = X
		l0 = l0/255
		l1 = sigmoid(np.dot(l0, h))
		#print(l1.T.shape)
		l2 = sigmoid(np.dot(l1.T, w))
		print(l2)

		l2_error = y-l2
		#if (i % 1000) == 0:
			#print(l2_error)
		'''
		l2_delta = l2_error*sigmoid(l2, deriv=True)
		l1_error = l2_delta.dot(w.T)
		l1_delta = l1_error*sigmoid(l1, deriv=True)

		w = l1.T.dot(l2_delta)
		h = l0.T.dot(l1_delta)
		'''
	print("-----------\n", l2)




if __name__ == '__main__':
	main()