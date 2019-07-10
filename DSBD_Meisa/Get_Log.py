
import matplotlib.pyplot as plt
import re

# with open('/home/C3D/C3D-v1.1/examples/c3d_ucf101_finetuning/TrainLog26.log') as f:
#     data = f.read()

with open('/home/C3D/C3D-v1.1/latest_result/logs/train7_1.log') as f:
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



plt.subplot(311)
plt.plot(iter_num[1:], test_loss[1:])
plt.subplot(312)
plt.plot(iter_num[1:], test_ac[1:])
plt.subplot(313)
plt.plot(train_iter_num[20:], train_loss[20:])

plt.show()