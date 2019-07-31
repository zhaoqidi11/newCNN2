
import matplotlib.pyplot as plt
import re
import numpy as np

def Get_Data(data):
    # pattern = re.compile(r'''
    # I0(.*?)solver.cpp:238]     Train net output #1: loss = (.*?) \(\* 1 = (.*?) loss\)
    # I0(.*?)sgd_solver.cpp:105] Iteration (.*?), lr =
    # ''')

    option = 10

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

    i = 0
    train_loss_ = 0

    for result in results1:
        if i%option == 0 and i!=0:
            train_iter_num.append(int(result[1]))
            train_loss.append(train_loss_/option)
            train_loss_ = 0
        else:
            train_loss_ += float(result[-1])
        i += 1


    for result in results2:
        iter_num.append(int(result[-3]))
        test_loss.append(float(result[3]))
        test_ac.append(float(result[1]))


    return {'train_iter_num':train_iter_num, 'train_loss':train_loss, 'test_iter_num':iter_num, 'test_loss':test_loss,'test_a':test_ac}

if __name__ == '__main__':

    with open('/home/C3D/C3D-v1.1/latest_result/models/bn_train_fewer/bn_train_fewer.log') as f:
        data1 = f.read()

    with open('/home/C3D/C3D-v1.1/latest_result/models/bn_train_pool_pad/bn_train_pool_pad_4.log') as f:
        data2 = f.read()

    with open('/home/C3D/C3D-v1.1/latest_result/models/deep_train_group_pool_pad/new/deep.log') as f:
        data3 = f.read()

    with open('/home/C3D/C3D-v1.1/latest_result/models/train_group_pool_pad/train_pool_pad.log') as f:
        data4 = f.read()


    results1 = Get_Data(data1)
    results2 = Get_Data(data2)
    results3 = Get_Data(data3)
    results4 = Get_Data(data4)

    option = 'train'
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(results1[option+'_iter_num'], results1[option+'_loss'], color='red', label='3D ResNet with BN')
    plt.plot(results2[option+'_iter_num'], results2[option+'_loss'], color='green', label='a variation of 3D ResNet with BN')
    plt.plot(results3[option+'_iter_num'], results3[option+'_loss'], color='dodgerblue', label='3D ResNet with GN')
    plt.plot(results4[option+'_iter_num'], results4[option+'_loss'], color='hotpink', label='a variation of 3D ResNet with GN')
    plt.xlabel('batches')
    plt.ylabel(option+' loss')

    plt.legend()

    plt.show()