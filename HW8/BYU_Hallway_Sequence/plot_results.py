import numpy as np
import matplotlib.pyplot as plt

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

    tx_data.append(tx)
    ty_data.append(ty)
    tz_data.append(tz)

tx_data = np.array(tx_data)
ty_data = np.array(ty_data)
tz_data = np.array(tz_data)

plt.figure(1)
plt.plot(tx_data, tz_data, 'r', label='estimated')
plt.legend()
plt.title('Task 2')
plt.xlabel('X distance in initial frame')
plt.ylabel('Z distance in initial frame')

plt.show()
