import numpy as np
import matplotlib.pyplot as plt
import struct
import socket
import time
import math

import rtde_control
import rtde_receive
import rtde_io

from ur5_sports import fead_forward
from ur5_sports import predict
from ur5_sports import Forward_Kinematics
from ur5_sports import Backward_Kinematics
from ur5_sports import Force_Compensation
from ur5_sports import Go_ahead
from ur5_sports import Up_ahead

IP_ADDR = '192.168.0.109'
PORT1 = 30005
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect((IP_ADDR,PORT1))
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
rtde_c = rtde_control.RTDEControlInterface("192.168.0.100")#控制信息

def cos(angle):
    return np.cos(angle)
def sin(angle):
    return np.sin(angle)
def atan2_math(y,x):
    return math.atan2(y,x)
def atan2(y,x):
    return np.arctan2(y,x)
def acos1(res):
    return np.arccos(res)
def sqrt(n):
    return np.sqrt(n)
def modify_S(x):
    if x >= 0:
        x = (x - (round(x)-1))*np.pi
    if x < 0:
        x = (round(x) - x)*np.pi
    return x
def modify(x):
    if x >= np.pi:
        x = x % np.pi
    if x < -np.pi:
        x = x % -np.pi
    return x
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

model = np.load(r"D:\Working park\model1.npy",allow_pickle=True)

def MBK(x,dx,ddx,M,B,K,dt,Fe):
    ddx1 = (Fe - B * dx - K * x) / M
    dx1 = dt * (ddx1 + ddx) / 2 + dx  # v1=dt*(a0+a1)/2+v0  加速度一次积分得速度
    x1 = dt * (dx1 + dx) / 2 + x  # 同理速度一次积分得位移
    return x1, dx1, ddx1

velocityJ = 0.08
acceleration = 0.01
dt = 1.0 / 1000  # 2ms
lookahead_time = 0.1
gain = 2000
################################
xe, dxe, ddxe = 0, 0, 0  # x轴方向力引起的位移，速度，加速度增量初始值均为0
ye, dye, ddye = 0, 0, 0
ze, dze, ddze = 0, 0, 0
Txe, dTxe, ddTxe = 0, 0, 0  # x轴方向力矩引起的位移，速度，加速度增量初始值均为0
Tye, dTye, ddTye = 0, 0, 0  # y轴方向力矩同理
Tze, dTze, ddTze = 0, 0, 0  # z轴方向力矩同理
fz = []
fy = []
px = []
py = []

list = []
for i in range(20):
    f = data_recv(s)
    if i >= 10:
        list.append(Force_Compensation(rtde_r.getActualQ(),f[2]))
print(list)
Fastening_Force = sum(list)/len(list)
joint_q = rtde_r.getActualQ()
j3 = joint_q[3]
j4 = joint_q[4]
d3j0 = 0

a = [0,-425.00,-392.25,0,0,0]#mm制
d = [89.159,0,0,109.15,94.65,82.30]
#a = [0,-0.42500,-0.39225,0,0,0]#m制
#d = [0.089159,0,0,0.10915,0.09465,0.08230]
alpha = [np.pi/2,0,0,np.pi/2,-np.pi/2,0]

while True:
    F = data_recv(s)
    joint_c = [0, 0, 0, 0, 0, 0]
    for i in range(6):
        joint_c[i] = joint_q[i] * 180 / np.pi
    FK_T = Forward_Kinematics(joint_q, d, a, alpha)
    print(F[2], Force_Compensation(joint_q, F[2]) - Fastening_Force)
    if Force_Compensation(joint_q, F[2]) - Fastening_Force < -1.5:
        res = predict(F, model)
        print("结果为：", res)
        if res == 1:  # My
            df = 0.0015 * (-1.5-Force_Compensation(joint_q, F[2]) - Fastening_Force)
            '''
            joint = rtde_r.getActualQ()
            d3j = round(joint[3] - j3,2)
            dd3j = d3j-d3j0
            x1,x2,x3 = MBK(joint[3],d3j,dd3j,M,B,K,dt,-1.5 - Force_Compensation(joint_q, F[2])-Fastening_Force)
'''
            joint_q[3] += df
            joint_q[4] += df
            '''
            j3 = joint[3]
            j4 = joint[4]
            d3j0 = d3j
            '''
            smash = 0
        if res == 4:
            df = 0.0015 * (-1.5 - Force_Compensation(joint_q, F[2]) - Fastening_Force)
            smash = 0
            joint_q[3] -= df
            joint_q[4] -= df
        if res == 3:
            df = 0.0015 * (-1.5 - Force_Compensation(joint_q, F[2]) - Fastening_Force)
            smash = 0
            joint_q[3] -= df
            joint_q[4] += df
        if res == 2:
            df = 0.0015 * (-1.5 - Force_Compensation(joint_q, F[2]) - Fastening_Force)
            smash = 0
            joint_q[3] += df
            joint_q[4] -= df
        if res == 5:
            df = 0.0015 * (-1.5 - Force_Compensation(joint_q, F[2]) - Fastening_Force)
            smash = 0
            joint_q[3] += df
        if res == 6:
            df = 0.0015 * (-1.5 - Force_Compensation(joint_q, F[2]) - Fastening_Force)
            smash = 0
            joint_q[3] -= df
        if res == 7:
            df = 0.0015 * (-1.5 - Force_Compensation(joint_q, F[2]) - Fastening_Force)
            smash = 0
            joint_q[4] += df
        if res == 8:
            df = 0.0015 * (-1.5 - Force_Compensation(joint_q, F[2]) - Fastening_Force)
            smash = 0
            joint_q[4] -= df
        if res == 9:
            smash += 1
            print("正碰撞!")
        if smash >= 50:
            Forward_Flag = False
            PID_Flag = True
            break

    else:
        # print("结果为：未碰撞")
        FK_T = FK_T.dot(Go_ahead(FK_T, 0.08))  # 获取往前推的新T矩阵
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
