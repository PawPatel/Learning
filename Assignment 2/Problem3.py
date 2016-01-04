import numpy as np 


def compute_function(X):
	return 100*(X[1] - X[0]**2)**2 + (1-X[0])**2


def compute_gradient(X):
	return np.array([400*X[0]**3 - 400*X[0]*X[1] + 2*X[0] - 2, -200*X[0]**2 + 200*X[1]])


def compute_hessian_inverse(X):
	det = 200*(1200*X[0]**2 - 400*X[1] + 2) - (400**2)*X[0]**2
	return (1/det)*np.array((200, 400*X[0]) , (400*X[0], 1200*X[0]**2 - 400*X[1] + 2) )

def main():
	numiterate = 100000
	stepsize = 1
	X = np.array([-1.2,1])
	method = 'newton'

	for i in range(0, numiterate):
		if method == 'gradient':
			X = X - stepsize*compute_gradient(X)
		else:
			X = X - stepsize*(np.dot(compute_hessian_inverse(X), compute_gradient(X)))

	print("After {0} iterations, the minimum is at {1} and is {2}").format(numiterate, X, compute_function(X))

if __name__ == '__main__':
    main()