import numpy as np


def sigmoid(x, deriv = False):
	if deriv == False :
		return (1/(1+np.exp(-x)))
	return x*(1-x)

def main():
	np.random.seed(0)
	learning_rate = 0.1
	X = np.array([[100, 255, 255],
				  [200, 150, 50],
				  [100, 100, 100],
				  [200, 200, 200]])
	Y = np.array([[1]])
	w1 = np.random.random((3, 12))
	w2 =  np.random.random((1, 3))

	for i in range(100000):
		l0 = X
		l0.shape = (12,1)
		l1 = sigmoid(np.dot(w1, l0))
		#print(l1.shape)
		l2 = sigmoid(np.dot(w2, l1))
		#print(l2.shape)

		l2_error = Y - l2
		#print(l2_error)
		l2_delta = sigmoid(l2, True)*l2_error
		#print(l2_delta)
		l1_error = np.sum(np.dot(l2, w2))
		#print(l1_error)
		l1_delta = sigmoid(l1, True)*l1_error
		#print(l1_delta)
		w1_change = learning_rate * l1_delta * X.T
		#print(w1_change)
		w2_change = learning_rate * l2_delta * l1.T
		#print(w2_change)
		w1 += w1_change
		w2 += w2_change

		if i % 100 == 0:
			print(l2)
	print(l2)

if __name__ == '__main__':
	main()