import socket

IP_ADDR = '192.168.71.197'
PORT1 = 2000
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
while(1):
    try:
        if(eval(str(PORT1))>10000):
            break
        s.connect((IP_ADDR, PORT1))
        print("获得端口：" + str(PORT1))

        break
    except Exception:
        print("当前正在访问IP:"+IP_ADDR + "   端口："+str(PORT1))
        PORT1 = PORT1 + 1

