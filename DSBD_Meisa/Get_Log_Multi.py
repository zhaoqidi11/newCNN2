
import matplotlib.pyplot as plt
import re
import numpy as np

def Get_Data(data):
    # pattern = re.compile(r'''
    # I0(.*?)solver.cpp:238]     Train net output #1: loss = (.*?) \(\* 1 = (.*?) loss\)
    # I0(.*?)sgd_solver.cpp:105] Iteration (.*?), lr =
    # ''')


    pattern1 = re.compile(r'''I0(.*?)solver.cpp:219] Iteration (.*?) \((.*?)\), loss = (.*?)\n''')

    # pattern2 = re.compile(r'''
    # I0(.*?)solver.cpp:331] Iteration (.*?), (.*?)
    # I0(.*?)solver.cpp:398]     Test net output #0: accuracy/top-1 = (.*?)
    # I0(.*?)solver.cpp:398]     Test net output #1: loss = (.*?) \((.*?)\)
    # ''')

    pattern2 = re.compile(r'''I0(.*?)solver.cpp:398]     Test net output #0: accuracy/top-1 = (.*?)\nI0(.*?)solver.cpp:398]     Test net output #1: loss = (.*?) \((.*?)\)\nI0(.*?)solver.cpp:219] Iteration (.*?) \((.*?)\), loss = (.*?)''')
    results1 = re.findall(pattern1, data)
    results2 = re.findall(pattern2, data)
    iter_num = []
    test_loss = []

    test_ac = []
    train_iter_num = []
    train_loss = []

    print(results2)

    for result in results1:
        train_iter_num.append(int(result[1]))
        train_loss.append(float(result[-1]))

    for result in results2:
        iter_num.append(int(result[-3]))
        test_loss.append(float(result[3]))
        test_ac.append(float(result[1]))

    return {'train_iter_num':train_iter_num, 'train_loss':train_loss, 'iter_num':iter_num, 'test_loss':test_loss,'test_a':test_ac}

if __name__ == '__main__':

    with open('/home/C3D/C3D-v1.1/latest_result/logs/new_train_group3.log') as f:
        data1 = f.read()

    with open('/home/C3D/C3D-v1.1/latest_result/logs/new_train_group6.log') as f:
        data2 = f.read()

    results1 = Get_Data(data1)
    results2 = Get_Data(data2)

    plt.plot(results1['iter_num'][3:], results1['test_loss'][3:], color='green', label='old')
    plt.plot(results2['iter_num'][3:], results2['test_loss'][3:], color='red', label='new')
    plt.legend()

    plt.show()