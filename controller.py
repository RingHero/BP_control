import numpy as np
import matplotlib.pyplot as plt
import struct
import socket
import time

import rtde_control
import rtde_receive
import rtde_io

from ur5_sports import Backward_Kinematics
from ur5_sports import Forward_Kinematics
from ur5_sports import Force_Compensation
from ur5_sports import Go_ahead

import random
import xlrd
import xlwt
from xlutils.copy import copy
from random import randint

def sigmoid(z):
    h = 1. / (1 + np.exp(-z))
    return h
def de_sigmoid(h):
    return h * (1 - h)
# 无激活函数（全1矩阵）前反向传递函数
def no_active(z):
    h = z
    return h
def de_no_active(h):
    return np.ones(h.shape)
# 推演函数
def fead_forward(datas, layers):
    input_layers = []

    for i in range(len(layers)):
        layer = layers[i]
        if i == 0:
            inputs = datas
            z = np.dot(inputs, layer["w"]) + layer["b"]
            h = layer["act_fun"](z)
            input_layers.append(inputs)
        else:
            inputs = h
            z = np.dot(inputs, layer["w"]) + layer["b"]
            h = layer["act_fun"](z)
            input_layers.append(inputs)
    output = np.argmax(h)+1
    return output
def predict(F,model):
    trait = np.array([F[3] / F[2], F[4] / F[2], F[5] / F[2]])
    res = fead_forward(trait, model)
    return res
def data_recv(recv):
    data = recv.recv(1000)
    fx = struct.unpack('f', data[6:10])[0]
    fy = struct.unpack('f', data[10:14])[0]
    fz = struct.unpack('f', data[14:18])[0]
    mx = struct.unpack('f', data[18:22])[0]
    my = struct.unpack('f', data[22:26])[0]
    mz = struct.unpack('f', data[26:30])[0]
    F = np.array([fx, fy, fz, mx, my, mz])
    return F

#正逆运动学参数
a = [0,-425.00,-392.25,0,0,0]#mm制
d = [89.159,0,0,109.15,94.65,82.30]
alpha = [np.pi/2,0,0,np.pi/2,-np.pi/2,0]

#加载模型
model = np.load(r"D:\Working park\model2.npy",allow_pickle=True)
#左上角点1右上角2左下角3右下角4上边5下边6左边7右边8
IP_ADDR = '192.168.0.109'
PORT1 = 30005
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect((IP_ADDR,PORT1))
#查看链接地址
sendData = "AT+EIP=?\r\n"
s.send(sendData.encode())
recvData = bytearray(s.recv(1000))
print(recvData)
#查看解耦矩阵
sendData = "AT+DCPMA6X8=?\r\n"
s.send(sendData.encode())
recvData = bytearray(s.recv(1000))
print(recvData)
#计算单位
sendData = "AT+DCPCU=?\r\n"
s.send(sendData.encode())
recvData = bytearray(s.recv(1000))
print(recvData)
#设置采样频率
set_update_rate = "AT+SMPF=100\r\n"
s.send(set_update_rate.encode())
recvData = bytearray(s.recv(1000))
print(recvData)
#矩阵运算单位
set_compute_unit = "AT+DCPCU=MVPV\r\n"
s.send(set_compute_unit.encode())
#上传数据格式
set_recieve_format = "AT+SGDM=(A01,A02,A03,A04,A05,A06);E;1;(WMA:1)\r\n"
s.send(set_recieve_format.encode())
recvData = bytearray(s.recv(1000))
print(recvData)
#连续上传数据包
get_data_stream = "AT+GSD\r\n"
s.send(get_data_stream.encode())
print("开始接收\r\n")

rtde_r = rtde_receive.RTDEReceiveInterface("192.168.0.100")#接收消息
rtde_c = rtde_control.RTDEControlInterface("192.168.0.100")
task_frame = [0,0,0,0,0,0]
selection_vector = [0,0,1,0,0,0]


#moveL路径改变TCP控制
velocityJ = 0.08
acceleration = 0.01
accelerationL = 0.005
dt = 1.0/1000
lookahead_time = 0.1
gain = 2000
velocityL = 0.005
df = 0.001
#rtde_c.moveL(path)
#rtde_c.stopScript()
#rtde_c.moveJ(joint_q,velocity,acceleration,0.0)

'''
for i in range(300):
    t_start = rtde_c.initPeriod()
    rtde_c.servoJ(joint_q, velocity, acceleration, dt, lookahead_time, gain)
    #当视角从后面看的时候（电机顺序从下往上）：
    #joint_q[0] += 0.001#最底部电机往左
    #joint_q[1] += 0.001#次底部电机往上
    #joint_q[2] += 0.001#第三电机往上
    #joint_q[3] += 0.001#从上往下数第三个电机往上（末端执行点抬头）
    #joint_q[4] += 0.001#从上往下数第二个电机往左（末端执行点扭头）
    #joint_q[5] += 0.001#末端执行点往右旋转
    rtde_c.waitPeriod(t_start)
'''

list = []
for i in range(20):
    f = data_recv(s)
    if i >= 10:
        list.append(Force_Compensation(rtde_r.getActualQ(),f[2]))
print(list)
Fastening_Force = sum(list)/len(list)
#rtde_c.servoStop()
#rtde_c.stopScript()

df = 0.001
joint_q = rtde_r.getActualQ()
smash = 0
Forward_Flag = True
Search_Flag = False
PID_Flag = False
while Forward_Flag:
    F = data_recv(s)
    FK_T = Forward_Kinematics(joint_q, d, a, alpha)
    print(F[2], Force_Compensation(joint_q, F[2]) - Fastening_Force)
    if Force_Compensation(joint_q,F[2]) - Fastening_Force < -1.8:
        res = predict(F, model)
        print("结果为：", res)
        if res == 1:  # My
            joint_q[3] += df
            joint_q[4] += df
            smash = 0
        if res == 4:
            joint_q[3] -= df
            joint_q[4] -= df
            smash = 0
        if res == 3:
            joint_q[3] -= df
            joint_q[4] += df
            smash = 0
        if res == 2:
            joint_q[3] += df
            joint_q[4] -= df
            smash = 0
        if res == 5:
            joint_q[3] += df
            smash = 0
        if res == 6:
            joint_q[3] -= df
            smash = 0
        if res == 7:
            joint_q[4] += df
            smash = 0
        if res == 8:
            joint_q[4] -= df
            smash = 0
        if res == 9:
            smash += 1
            print("正碰撞!")
        if smash >= 100:
            Forward_Flag = True
            PID_Flag = False
            #Search_Flag = True
    else:
        FK_T = FK_T.dot(Go_ahead(FK_T, 0))  # 获取往前推的新T矩阵
        # print(FK_T[0][3],FK_T[1][3],FK_T[2][3])
        BK_T = Backward_Kinematics(FK_T)
        # print(BK_T)
        # print(joint_q)
        for i in range(8):
            right = 0
            for j in range(6):
                if abs(BK_T[i][j] - joint_q[j]) < 0.6:
                    right += 1
            if right == 5:
                joint_q = [BK_T[i][0], BK_T[i][1], BK_T[i][2], BK_T[i][3], BK_T[i][4], joint_q[5]]
                # print("新矩阵已产生")
                break

    rtde_c.servoJ(joint_q, velocityJ, acceleration, dt, lookahead_time, gain)
    #print(F)

while PID_Flag:
    F = data_recv(s)
    FK_T = Forward_Kinematics(joint_q, d, a, alpha)
    F_modify = Force_Compensation(joint_q, F[2]) - Fastening_Force
    print(F[2], F_modify)
    if F_modify >= -1.0 and F_modify <= -0.6:#期望范围
        print("标准碰撞")
        Search_Flag = True
    if F_modify < -1.0:#过推
        FK_T = FK_T.dot(Go_ahead(FK_T,-0.01))
    if F_modify >-0.6:#欠推
        FK_T = FK_T.dot(Go_ahead(FK_T,0.01))

while Search_Flag:
    break
