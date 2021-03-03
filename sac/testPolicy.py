import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from TaesEnvLander import LanderEnv
import pandas as pd
import numpy as np

numTest=10
actor=torch.load('policy.pkl')
env=LanderEnv()
maxSteps=int(env.Tf/env.dt)

for i in range(numTest):
    plt.figure(i)
    t=[]
    vx=[]
    vy=[]
    vz=[]
    s=env.reset()
    t.append(env.t)
    vx.append(s[0]*10)
    vy.append(s[1]*10)
    vz.append(s[2]*10)
    count=0
    while True:
        count+=1
        with torch.no_grad():
            a,_=actor(torch.as_tensor(s, dtype=torch.float32),True,False)
            a=a.numpy()
        s,_,d,_=env.step(a)
        t.append(env.t)
        vx.append(s[0]*10)
        vy.append(s[1]*10)
        vz.append(s[2]*10)
        if d or count==maxSteps:
            break    
    
    plt.plot(t,vx)
    plt.plot(t,vy)
    plt.plot(t,vz)
    plt.show()

# plt.figure(2)
# df_news = pd.read_table('Reward.txt',header = None)
# avr_len=50
# ptr=0
# totalEp=len(df_news[0])
# avr_score=np.zeros((totalEp))
# episode=np.zeros((totalEp))
# for i in range(totalEp):
#     episode[i]=i
#     lower=max(0,i-avr_len+1)
#     upper=i
#     avr_score[i]=sum(df_news[1][lower:(upper+1)])/(upper-lower+1)
# # plt.plot(df_news[0],df_news[1])
# plt.plot(episode,avr_score)
# plt.xlabel('step')
# plt.ylabel('Average Episode reward')
# plt.show()