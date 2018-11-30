import numpy as np


def back_prop(w):
    error = learn_rate*(y - y_pred)
        
    w[0] = w[0] + error*x[0]
    w[1] = w[1] + error*x[1]
    w[2] = w[2] + error*x[2]
    w[3] = w[3] + error*x[3]

    return w

def main():
    w0 = 0.7
    w1 = -0.2
    w2 = 0.1
    w3 = 0.9
    w = np.array([w0, w1, w2, w3])
    learn_rate = 0.1

    x = [1, 0, 1, 0]
    y = 1
    # for i in range(x.shape[0]):
    #     print(x[i], y[i])

    #training

    for i in range(1000):
        y_pred = w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + w[3]*x[3]
        #print(f"y_pred {y_pred}")
        w = backprop(w)

    node = w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + w[3]*x[3]
    print(node)
    if node > 0:
        print(1)
    else:
        print(-1)


if __name__ == '__main__':
    main()
