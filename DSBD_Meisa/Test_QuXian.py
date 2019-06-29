
import matplotlib.pyplot as plt
import re
import numpy as np

# with open('/home/C3D/C3D-v1.1/examples/c3d_ucf101_finetuning/TrainLog26.log') as f:
#     data = f.read()

with open('/home/C3D/C3D-v1.1/newdsbd/new_train_log_20.log') as f:
    data = f.read()
# pattern = re.compile(r'''
# I0(.*?)solver.cpp:238]     Train net output #1: loss = (.*?) \(\* 1 = (.*?) loss\)
# I0(.*?)sgd_solver.cpp:105] Iteration (.*?), lr =
# ''')

pattern1 = re.compile(r'''
I0(.*?)solver.cpp:219] Iteration (.*?) \((.*?)\), loss = (.*?)
''')

# pattern2 = re.compile(r'''
# I0(.*?)solver.cpp:331] Iteration (.*?), (.*?)
# I0(.*?)solver.cpp:398]     Test net output #0: accuracy/top-1 = (.*?)
# I0(.*?)solver.cpp:398]     Test net output #1: loss = (.*?) \((.*?)\)
# ''')

pattern2  = re.compile(r'''
I0(.*?)solver.cpp:398]     Test net output #0: accuracy/top-1 = (.*?)
I0(.*?)solver.cpp:398]     Test net output #1: loss = (.*?) \((.*?)\)
I0(.*?)solver.cpp:219] Iteration (.*?) \((.*?)\), loss = (.*?)
''')

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

average_test = []
# Count = 73

AllNum = 50
Count = len(test_loss) - AllNum
for i in range(Count):
    average_test.append(np.average(test_loss[i:i+AllNum]))

average_x_axis = range(Count)


average_train = []

# train_num = 432
train_num = 2000
train_count = len(train_loss) - train_num


for i in range(train_count):
    average_train.append(np.average(train_loss[i:i+train_num]))


average_x_axis_train = range(train_count)

plt.subplot(211)
plt.plot(average_x_axis, average_test)

plt.subplot(212)
plt.plot(average_x_axis_train, average_train)


# plt.subplot(311)
# plt.plot(iter_num, test_loss)
# plt.subplot(312)
# plt.plot(iter_num, test_ac)
# plt.subplot(313)
# plt.plot(train_iter_num, train_loss)

plt.show()