import numpy as np


def main():
    w1 = round(np.random.random(), 2)
    w2 = round(np.random.random(), 2)
    w3 = round(np.random.random(), 2)
    w4 = round(np.random.random(), 2)

    print(w1, w2, w3, w4)

    x = np.array([[1, 1, 1, 1],
                  [0, 0, 0, 0],
                  [1, 1, 0, 0],
                  [1, 1, 1, 0],
                  [1, 0, 0, 0]])
    y = np.array([[1],
                  [-1],
                  [0],
                  [1],
                  [0]])

    for i in range(x.shape[0]):
        print(x[i], y[i])


if __name__ == '__main__':
    main()
