#!/usr/bin/env python
# coding: utf-8

# In[16]:


import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
df = pd.read_csv ('simulation.csv')
print(df)


# In[2]:


def SenDySched(Te,CL):
    Tes = np.sort(Te)
    #print(Tes)
    Tsp = np.min(Tes)
    #print(Tsp)
    L1 = Tsp
    L2 = len(Tes)
    for i in range(L1):
        for j in Tes:
            n = np.ceil(j/Tsp)
            thresh = j + CL
            #print(j,n,Tsp,n*Tsp,thresh)
            if n*Tsp > thresh:
                Tsp -= 1
                break
    if Tsp > 1:
        return Tsp
    else:
        print("alg. failed")  

def gen_rand_Te(HL):
    none = random.sample(range(50, 200), HL*HL+1)
    microwave = random.sample(range(30, 300), HL)
    kettle = random.sample(range(50, 240), HL)
    faucet = random.sample(range(15, 120), HL)
    wdisposer = random.sample(range(10, 60), HL)
    vfan = random.sample(range(200, 600), HL)
    events_Te = [none,microwave,kettle,faucet,wdisposer,vfan]
    return events_Te
def sim_plot(sim,eclass):    
    plot_a = []
    L = len(sim)
    for i in range(L):
        for j in range(sim[i]):
            c = eclass[i]
            #print(j,c)
            plot_a.append([int(c)])
    return plot_a
def delay(current_event, test_events, position):
    t = 0
    while current_event != test_events[position]:
        position -= 1
        t += 1
    return t


# In[3]:


class train_Qsched:
    def __init__(self, Time,events,Tsp,CL,position,n_actions,test):
        #self.size = size
        self.position = position
        self.event_att = 0
        self.Time = Time
        self.length = len(Time)
        self.events = events
        self.Tsp = Tsp
        self.Tspold = Tsp
        self.CL = CL
        self.reward = 0
        self.Tideal = 0
        self.num_classes = 6
        self.cur_event = [0]
        self.pre_event = [0]
        self.event_states = 2
        self.num_states = self.num_classes*n_actions
        #self.num_states = self.num_classes*self.event_states
        self.num_actions = n_actions#0
        self.increment = 1 #or 5 or 10
        self.event_changed = 0 #0=no, 1=yes
        self.w1 = 1
        self.w2 = 1
        self.step = 0
        self.done = False
        self.test = test
    def get_qtable(self):
        self.qtable = np.zeros([self.num_states, self.num_actions])
        return self.qtable
    
    def update_qtable(self, new_qtable):
        self.qtable = new_qtable
        
    def reset_position(self):
        self.position = 0
        
    def take_action(self, action):
        Tsp = action+1
        #print(Tsp)
        reward = 0
        inc = self.increment
        done = self.done
        self.pre_event = self.events[self.position]
        self.step += 1
        self.position += Tsp
        #print(self.position)
        self.Tsp = Tsp
        #print(self.cur_event)
        cl = self.CL[self.cur_event[0]]
        if self.position + Tsp > self.length: 
            done = True
            #print("done",self.position,self.length)
            self.position = self.length - 1
            
        Tideal = self.find_Tideal(self.events,self.position)
        #print(Tsp,self.Tspold)
        if Tsp >= Tideal:
            if Tsp - Tideal <= cl:
                reward = 50
            else:
                reward = -50#*self.w1
        #elif Tsp < Tideal:
        #    if Tideal - Tsp < cl:
        #        reward = 0
        #    else:
        #        reward = -10*self.w2 
            #reward = -5
            
            #returnhere

        else:
            if Tsp >= self.Tspold:
                reward = 30
            else: 
                reward = -30         
        if self.events[self.position] == self.cur_event:
            self.event_changed = 0
            #reward += -5
        else:
            self.cur_event = self.events[self.position] 
            self.event_changed = 1
            #reward += 50
            #print(self.cur_event,self.event_changed)
        #print(reward)
        #next_state = self.encode0(self.cur_event[0],self.event_changed)
        i = (self.cur_event[0])*self.num_actions
        #next_Tsp = np.max(np.where(self.qtable==self.qtable[i:i+self.num_actions-1].max()))#[0][0]#-i
        #next_Tsp = np.where(self.qtable[i:i+self.num_actions]==self.qtable[i:i+self.num_actions].max())[0][0]
        next_Tsp = random.randint(0, self.num_actions-1)
        #print(self.cur_event[0],Tsp,next_Tsp)
        if self.test:
            next_state = self.encode(self.cur_event[0],Tsp)
        else:
            next_state = self.encode(self.cur_event[0],Tsp)#next_Tsp)
        

        
        self.Tspold = Tsp
        '''
        print("event cathced")
        cposition_arr.append([position])
        cevent_hist.append(event_cur)
        correct += 1
        print("Ideal Tsp:", find_Tideal(events,position,CL))
        else:
        print("missed!!")
        missed += 1
        xposition_arr.append([position])
        xevent_hist.append(event_cur)
        event_cur = events[position]
        '''
        return next_state, reward, done, self.cur_event[0]
    
    def get_info(self):
        return self.position,self.pre_event,self.event_changed
    
    def encode0(self, eclass,changed):
        if changed == 0:
            i = eclass
        else:
            i = eclass+self.num_classes
        return i
    
    def encode(self, eclass,Tsp):
        # (5) 5, 5, 4
        i = (eclass)*self.num_actions#*eclass
    
        #print(i)
        #i *= changed+1
        ##i *= 6
        '''
        if changed == 0:
            i *= 1
        else:
            i *= 2
            i += self.num_actions
        '''
        i += Tsp - 1
        return i
    
    def find_Tideal(self, events,position):
        new_position = position
        #print("position:",position)
        current_event = events[position]
        #print(current_event)
        #l = len(events)-position
        #cl = CL[current_event[0]]
        #Tideal = 0
        while current_event == events[new_position] and (new_position < self.length - 1):
            new_position += 1
        Tideal = new_position-position
        #print(Tideal,self.length)
        return Tideal# + cl/2
    
    def print_info(self):
        print("current Tsp:",self.Tsp)
        print("current Position:",self.position)
        print("current event changed?",self.event_changed)
        


# In[ ]:





# In[4]:


HL=5
test_events_Te = gen_rand_Te(HL)


test_events_seq = []
test_seq_periods = []
xo = random.randint(1, 5)
yo = random.randint(0, 4)
for none in test_events_Te[0]:
    test_events_seq.append([0])
    test_seq_periods.append([none])
    x = random.randint(1, 5)
    y = random.randint(0, 4)
    if x == xo:
        x = random.randint(1, 5)
    xo = x
    test_events_seq.append([x])
    test_seq_periods.append([test_events_Te[x][y]])
test_events_seq.append([0])
test_seq_periods.append([random.randint(50, 200)])
#print(np.hstack(test_events_seq))
#print(np.hstack(test_seq_periods))

test_events = sim_plot(np.hstack(test_seq_periods),np.hstack(test_events_seq))
l = len(test_events)
test_time = np.linspace(0, l, num=l, endpoint=True)


# In[5]:


CL = [5,12,15,5,8,15]
Tsp = random.randint(0,10)
position = 0
n_actions = 100
#ql_sched_test = train_Qsched(test_time,test_events,Tsp,CL,position,n_actions,True)
done = False
itera = 0
q_table = np.load('q_table.npy')


# In[32]:


total_epochs, total_penalties = 0, 0
episodes = 1
state = 0
CL = [5,12,15,5,8,15]
Tsp = random.randint(0,50)
position = 0
done = False
event = 0
pens = 0 
steps = 0
ql_sched_test = train_Qsched(test_time,test_events,Tsp,CL,position,n_actions,True)
ql_sched_test.update_qtable(q_table)
#for _ in range(episodes):
total_time = 0    
ql_exec_time = []
print("QL OVERHEAD")

for i in range(episodes):
    done = False
    epochs, penalties, reward = 0, 0, 0

    while not done:
        stime = time.time()
        action = np.argmax(q_table[state])
        #print("Tsp: ",action+1," cur event: ",event," state: ",state)
        state, reward, done, event = ql_sched_test.take_action(action)
        pos, pe, changed  = ql_sched_test.get_info()
        #print(pos, ce,changed, "reward:", reward)
        if changed == 1:
            if delay(pe, test_events, pos) > CL[pe[0]]:
                pens += 1
            #print(delay(pe, test_events, pos))
            #print(test_events[pos],pe)
        etime = time.time()
        if reward <= -10:
            penalties += 1
        ql_exec_time.append(abs(etime-stime))
        epochs += 1
        steps += 1
        total_time += abs(etime-stime)
    ql_sched_test.reset_position()
    total_penalties += penalties
    total_epochs += epochs

#print(f"Results after {episodes} episodes:")
#print(f"Average timesteps per episode: {total_epochs / episodes}")
#print(f"Average penalties per episode: {total_penalties / episodes}")
#print(f"Steps: {steps}/{l}, Penalties: {pens}")
#print(f"Average execution time per step: {total_time / (episodes*steps)}")


# In[33]:


print("QL mean",statistics.mean(ql_exec_time))
print("QL min",min(ql_exec_time))
print("QL max",max(ql_exec_time))
print("QL std",statistics.stdev(ql_exec_time))


# In[7]:


event_Tsp = np.load("event_Tsp.npy")


# In[30]:


print("STATIC OVERHEAD")
current_Tsp = event_Tsp[0]
current_CL = CL[0]
current_event = [0]
position = 0
length = len(test_events)
penalty = 0
steps = 0
avg_Tsp = 0
total_time = 0
st_exec_time = []
while length > 0:
    stime = time.time()
    #print(current_event,current_Tsp,current_CL)
    if current_event != test_events[position]:
        if delay(current_event, test_events, position) > current_CL:
            penalty += 1
        #print(current_event,test_events[position],position,penalty)
        current_event = test_events[position]
        current_Tsp = event_Tsp[current_event[0]]
        current_CL = CL[current_event[0]]
    position += current_Tsp[0]
    length -= current_Tsp[0]
    steps += 1
    etime = time.time()
    st_exec_time.append(abs(etime-stime))
    total_time += abs(etime-stime)
    #print(position)
#print(f"Steps: {steps}/{l}, Penalties: {penalty}")
#print(f"Average execution time per step: {total_time / (episodes*steps)}")


# In[31]:


print("STATIC  mean",statistics.mean(st_exec_time))
print("STATIC  min",min(st_exec_time))
print("STATIC  max",max(st_exec_time))
print("STATIC  std",statistics.stdev(st_exec_time))


# In[ ]:




