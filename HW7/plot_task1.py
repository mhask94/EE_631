import numpy as np
import matplotlib.pyplot as plt

data_file = open('task1_data.txt')

frame = []
tau = []

for line in data_file.readlines(): 
    data = line.split('\t') 
    fr = float(data[0])
    t = float(data[1])

    frame.append(fr) 
    tau.append(t) 

frame = np.array(frame)
tau = np.array(tau)

# batch LS 1st order
A = np.vstack([frame, np.ones(len(frame))]).T
b = np.vstack([tau]).T
c = np.linalg.inv(A.T @ A) @ A.T @ b

x_hit = -c.item(1)/c.item(0)
x = np.linspace(0,x_hit)
y = c.item(0)*x + c.item(1)

hit_label = 'impact at ' + str(x_hit)[:5] + ' frames'

plt.figure(1)
plt.plot(x,y,'r',label='fit')
plt.plot(frame,tau,'b.',label='truth')
plt.plot(x_hit,0,'k+',label=hit_label)
plt.xlabel('Frame Number')
plt.ylabel('Estimated Frames to Impact')
plt.xlim([-1,20])
plt.legend()
plt.title('Task 1')

plt.show()
