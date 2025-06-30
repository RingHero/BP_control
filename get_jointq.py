import numpy as np
import matplotlib.pyplot as plt
import struct
import socket

import rtde_control
import rtde_receive
import rtde_io


rtde_r = rtde_receive.RTDEReceiveInterface("192.168.0.100")#接收消息
rtde_c = rtde_control.RTDEControlInterface("192.168.0.100")

velocity = 0.02
acceleration = 0.04
dt = 1.0/500
lookahead_time = 0.1
gain = 300

for i in range(1500):
    actualQ = rtde_r.getActualQ()
    print(actualQ)
    t_start = rtde_c.initPeriod()
    actualQ[3] += 0.004
    rtde_c.servoJ(actualQ, velocity, acceleration, dt, lookahead_time, gain)
    rtde_c.waitPeriod(t_start)


rtde_c.servoStop()
rtde_c.stopScript()