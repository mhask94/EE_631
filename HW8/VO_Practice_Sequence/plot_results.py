import numpy as np
import matplotlib.pyplot as plt
from IPython.core.debugger import Pdb

true_file = open('truth.txt')

tx_true = []
ty_true = []
tz_true = []

for line in true_file.readlines():
    data = line.split(' ')
    if len(data) < 12:
        continue
    _ = data[0]
    _ = data[1]
    _ = data[2]
    tx = float(data[3])
    _ = data[4]
    _ = data[5]
    _ = data[6]
    ty = float(data[7])
    _ = data[8]
    _ = data[9]
    _ = data[10]
    data = data[11].split('\n')
    tz = float(data[0])

    tx_true.append(tx)
    ty_true.append(ty)
    tz_true.append(tz)

data_file = open('data.txt')

tx_data = []
ty_data = []
tz_data = []

for line in data_file.readlines():
    data = line.split('\t')
    _ = data[0]
    _ = data[1]
    _ = data[2]
    tx = float(data[3])
    _ = data[4]
    _ = data[5]
    _ = data[6]
    ty = float(data[7])
    _ = data[8]
    _ = data[9]
    _ = data[10]
    tz = float(data[11])
#    Pdb().set_trace()

    tx_data.append(tx)
    ty_data.append(ty)
    tz_data.append(tz)

tx_true = np.array(tx_true)
ty_true = np.array(ty_true)
tz_true = np.array(tz_true)
tx_data = np.array(tx_data)
ty_data = np.array(ty_data)
tz_data = np.array(tz_data)

plt.figure(1)
plt.plot(tx_true, tz_true, 'k', label='truth')
plt.plot(tx_data, tz_data, 'r', label='estimated')
plt.legend()
plt.title('Task 1')

plt.show()
