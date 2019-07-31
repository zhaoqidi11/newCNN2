
import matplotlib.pyplot as plt
import re
import numpy as np



if __name__ == '__main__':

    x = [0, 10000, 20000, 30000, 40000, 45000]

    y1 = [87.33, 4.22227, 2.42514, 0.933971, 0.937489, 0.931111]

    y2 = [87.33, 3.13992, 3.75814, 1.76295, 1.15464, 1.10061]

    y3 = [1.86944, 0.669116, 0.642168, 0.597614, 0.576278, 0.581749]

    y4 = [2.03, 0.515302, 0.576533, 0.588595, 0.583517, 0.58449]




    option = 'test'
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(x, y1, color='red', label='3D ResNet with BN')
    plt.plot(x, y2, color='green', label='a variation of 3D ResNet with BN')
    plt.plot(x, y3, color='dodgerblue', label='3D ResNet with GN')
    plt.plot(x, y4, color='hotpink', label='a variation of 3D ResNet with GN')
    plt.xlabel('batches')
    plt.ylabel(option+' loss')

    plt.legend()

    plt.show()