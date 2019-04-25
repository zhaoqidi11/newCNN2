
import matplotlib.pyplot as plt
import re

with open('/media/user02/Volume/C3D/C3D-v1.1/examples/c3d_ucf101_finetuning/train_log5.log') as f:
    data5 = f.read()

with open('/media/user02/Volume/C3D/C3D-v1.1/examples/c3d_ucf101_finetuning/train_log6.log') as f:
    data6 = f.read()

with open('/media/user02/Volume/C3D/C3D-v1.1/examples/c3d_ucf101_finetuning/train_log8.log') as f:
    data8 = f.read()

with open('/media/user02/Volume/C3D/C3D-v1.1/examples/c3d_ucf101_finetuning/train_log10.log') as f:
    data10 = f.read()

with open('/media/user02/Volume/C3D/C3D-v1.1/examples/c3d_ucf101_finetuning/train_log11.log') as f:
    data11 = f.read()

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

results5 = re.findall(pattern1, data5)
results5_ = re.findall(pattern2, data5)

results6 = re.findall(pattern1, data6)
results6_ = re.findall(pattern2, data6)

results8 = re.findall(pattern1, data8)
results8_ = re.findall(pattern2, data8)

results10 = re.findall(pattern1, data10)
results10_ = re.findall(pattern2, data10)

results11 = re.findall(pattern1, data11)
results11_ = re.findall(pattern2, data11)


iter_num = []
test_loss = []

test_ac = []
train_iter_num = []
train_loss = []



for result in results5:
    if int(result[1])>=2000:
        continue
    train_iter_num.append(int(result[1]))
    train_loss.append(float(result[-1]))


for result in results5_:
    if int(result[-3])>=2000:
        continue
    iter_num.append(int(result[-3]))
    test_loss.append(float(result[3]))
    test_ac.append(float(result[1]))

for result in results6:
    if int(result[1])>=8000:
        continue
    train_iter_num.append(int(result[1]))
    train_loss.append(float(result[-1]))


for result in results6_:
    if int(result[-3])>=8000:
        continue
    iter_num.append(int(result[-3]))
    test_loss.append(float(result[3]))
    test_ac.append(float(result[1]))

for result in results8:
    if int(result[1])>=2000:
        continue
    train_iter_num.append(int(result[1])+8000)
    train_loss.append(float(result[-1]))


for result in results8_:
    if int(result[-3])>=2000:
        continue
    iter_num.append(int(result[-3])+8000)
    test_loss.append(float(result[3]))
    test_ac.append(float(result[1]))

for result in results10:
    if int(result[1])>=4000:
        continue
    train_iter_num.append(int(result[1])+10000)
    train_loss.append(float(result[-1]))


for result in results10_:
    if int(result[-3])>=4000:
        continue
    iter_num.append(int(result[-3])+10000)
    test_loss.append(float(result[3]))
    test_ac.append(float(result[1]))

for result in results11:
    if int(result[1])>=2000:
        continue
    train_iter_num.append(int(result[1])+14000)
    train_loss.append(float(result[-1]))


for result in results11_:
    if int(result[-3])>=2000:
        continue
    iter_num.append(int(result[-3])+14000)
    test_loss.append(float(result[3]))
    test_ac.append(float(result[1]))

plt.subplot(311)
plt.plot(iter_num, test_loss)
plt.subplot(312)
plt.plot(iter_num, test_ac)
plt.subplot(313)
plt.plot(train_iter_num, train_loss)

plt.show()