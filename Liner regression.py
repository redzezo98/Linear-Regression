import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_linear.csv').values     # đọc file
N = data.shape[0]
x = data[:,0].reshape(-1,1)  # diện tích nhà --- ma trận X
y = data[:,1].reshape(-1,1)  # Giá nhà --- ma trận Y
print(N)
print(data.shape)
print(x)
print('===========================================')
print(y)
plt.scatter(x,y)
plt.xlabel('mét vuông')
plt.ylabel('giá')


# Xếp chồng ngang
x = np.hstack((np.ones((N,1)),x)) # ====> Ma trận X



w = np.array([0.,1]).reshape(-1,1) # ====> tạo ma trận w  là [0,1] ==> chuyển vị


numOfIteration = 100

cost = np.zeros((numOfIteration,1)) # => Tạo mảng 100 x 1 , toàn số 0
learning_rate = 0.000001
for i in range(1,numOfIteration): # chạy từ 1 -> 100
    r = np.dot(x,w) - y # r = ma trận X * ma trận W - ma trận Y
    cost[i] = 0.5*np.sum(r*r)  # tổng các phần tử trong mảng
    w[0] -= learning_rate*np.sum(r) # w0 - đạo hàm của J theo w0

    w[1] -= learning_rate*np.sum(np.multiply(r,x[:,1].reshape(-1,1))) # w1 - đạo hàm của J theo w1
    print(cost[i])

predict = np.dot(x,w) # Nhân ma trận X với tham số w vừa tìm ra sau khi gradient descent ==> hàm y = wx

plt.plot((x[0][1],x[N-1][1]),(predict[0],predict[N-1]),'r') # vẽ đồ thị
plt.show()
    
x1 = 50
y1 = w[0] +w[1]*x1
print('Giá nhà cho 50 m^2 là: ',y1)