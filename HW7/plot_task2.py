import numpy as np
import matplotlib.pyplot as plt

data_file = open('task2_data.txt')

frame = []
mm = []

for line in data_file.readlines(): 
    data = line.split('\t') 
    fr = float(data[0])
    t = float(data[1])

    frame.append(fr) 
    mm.append(t) 

frame = np.array(frame)
mm = np.array(mm)

# batch LS 1st order
A = np.vstack([frame, np.ones(len(frame))]).T
b = np.vstack([mm]).T
c = np.linalg.inv(A.T @ A) @ A.T @ b

x_hit = -c.item(1)/c.item(0)
x = np.linspace(0,x_hit)
y = c.item(0)*x + c.item(1)

hit_label = 'impact at ' + str(x_hit)[:5] + ' frames'

plt.figure(1)
plt.plot(x,y,'r',label='fit')
plt.plot(frame,mm,'b.',label='truth')
plt.plot(x_hit,0,'k+',label=hit_label)
plt.xlabel('Frame Number')
plt.ylabel('Estimated Distance to Impact (mm)')
plt.xlim([-1,20])
plt.legend()
plt.title('Task 2')

plt.show()
