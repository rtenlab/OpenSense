#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import time as tm
#df = pd.read_csv ('simulation.csv')


# In[2]:


HL = 5 #history length
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
#print(none,microwave,kettle,faucet,wdisposer,vfan)


# In[3]:


def Check_Tsp(Te,CL,Tsp):
    Tes = np.sort(Te)
    #print(Tes)
    #print(Tsp)
    valid = False
    for t in Tes:
        #print(t)
        n = np.ceil(t/Tsp)
        #thresh = t + cl
        if n*Tsp-t <= CL:
            valid = True
            
            #print(n,n*Tsp,t,cl,'no')
            #print("Tsp not valid")
        else:
            valid = False
            break
            #print(n,n*Tsp,t,cl,'yes')
            #print("Tsp is valid for CL=",cl) 
        #print(valid)
    return valid


# In[4]:


def gen_timeline():
    events_seq = []
    seq_periods = []
    xo = random.randint(1, 5)
    yo = random.randint(0, 4)
    for none in events_Te[0]:
        events_seq.append([0])
        seq_periods.append([none])
        x = random.randint(1, 5)
        y = random.randint(0, 4)
        if x == xo:
            x = random.randint(1, 5)
        xo = x
        events_seq.append([x])
        seq_periods.append([events_Te[x][y]])
    events_seq.append([0])
    seq_periods.append([random.randint(50, 200)])
    return events_seq,seq_periods


# In[5]:


def sim_plot(sim,eclass):    
    plot_a = []
    L = len(sim)
    for i in range(L):
        for j in range(sim[i]):
            c = eclass[i]
            #print(j,c)
            plot_a.append([int(c)])
    return plot_a


# In[6]:


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
        
def encode(eclass,Tsp):
    # (5) 5, 5, 4
    i = (eclass)*50#*eclass
    
    print(i)
    #i *= changed+1
    ##i *= 6

    i += Tsp - 1 
    #i *= 1
    return i
def encode0(eclass,changed):
    # (5) 5, 5, 4
    if changed == 0:
        i = eclass
    else:
        i = eclass+6
    return i


# In[7]:


events_Te = np.load("events_Te.npy",allow_pickle=True)


# In[8]:




events_seq,seq_periods = gen_timeline()
tn_events = sim_plot(np.hstack(seq_periods),np.hstack(events_seq))
l = len(tn_events)
times = np.linspace(0, l, num=l, endpoint=True)
print(np.hstack(events_seq))
print(np.hstack(seq_periods))


# In[9]:


#time to update class-based sched
st = tm.time()
l = len(events_Te)
CL = [5,12,15,5,8,15]
num_CLs = 10
event_CL = []
event_Tsp = []
for i in range(l):
    event_Tsp.append([])
    #CL = np.min(events_Te[i])
    #CLs = np.sort(random.sample(range(1, CL+1), num_CLs))
    #event_CL.append(np.sort(CLs))
    #for j in range(num_CLs):
    event_Tsp[i].append(SenDySched(events_Te[i],CL[i]))
#print("Random selected CLs for each event:",event_CL)
cb_time = tm.time()-st
print(cb_time)

lines = ("Updating CLass-based_sched time = ", str(cb_time))
with open('sched_updating_time.txt', 'w') as f:
    for line in lines:
        f.write(line)
    f.write('\n')


# In[10]:


thresh_list = [0.1,0.01,0.001,0.0001]
upd_times = []
q_tables = []
for thresh in thresh_list:
    gen_timeline()
    print(np.hstack(events_seq))
    print(np.hstack(seq_periods))
    q_table = np.load('q_table.npy')
    CL = [5,12,15,5,8,15]
    Tsp = random.randint(0,10)
    position = 0
    n_actions = 100

    #%%time
    """Training the agent"""

    import random
    from IPython.display import clear_output
    position = 0
    sttime = tm.time()
    ql_sched = train_Qsched(times,tn_events,Tsp,CL,position,n_actions,False)
    # Hyperparameters
    alpha = 0.3#1
    gamma = 0.3#6
    epsilon = 0.1#1
    penalties = 0
    rewards = 0
    # For plotting metrics
    all_epochs = []
    all_penalties = []
    MAXitra = 50000
    changed = 0
    #thresh = 0.01#005#01
    conv_counter = 0
    pre_pen = 0
    cur_pen = 0
    for i in range(1, MAXitra):
        state = 0#random.randint(0,11)
        epochs, reward, = 0, 0
        done = False
        #print("OK")
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, n_actions-1) # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values
            #print(f"Action: {action}")
            #print("OKact")
            next_state, reward, done, cur_event = ql_sched.take_action(action)
            #print(done)
            #ql_sched.print_info()
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            #print(next_state)
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value
            #clear_output(wait=True)
            #print("OKqta")
            if reward <= -10:
                penalties += 1

            if reward >= 10:
                rewards += 1

            #print("OKrew")
            epochs += 1
            #clear_output(wait=True)
            #with np.printoptions(threshold=sys.maxsize):
            #print(q_table)
            #print("state",state)
            ##print("next_state",next_state)
            pre_pen = cur_pen

            if state != next_state:
                changed += 1
            #print("OKstate")
            if i % 100 == 0:
                clear_output(wait=True)

                print(f"Episode: {i}")
                print(f"Action: {action}")
                print(f"reward: {reward}")
                print("pens: ", penalties/i)
                print("r: ", rewards/i)
                print(f"current state: {state}, next state: {next_state}, CHANGED: {changed}")
                print("conv counter:", conv_counter)
                ql_sched.print_info()



            state = next_state
            ql_sched.update_qtable(q_table)
        cur_pen = penalties/i    
        #print("conv counter:", conv_counter)
        #print("pre_pen = cur_pen",  pre_pen - cur_pen)
        if abs(pre_pen - cur_pen) <= thresh:
            conv_counter += 1
        else:
            conv_counter = 0
        if conv_counter >= 10:
            break    
        ql_sched.reset_position()
        '''
        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")
            print(f"Action: {action}")
            print(f"reward: {reward}")
            ql_sched.print_info()
        '''
    print("Training finished.\n")
    upd_times.append(tm.time()-sttime)
    q_tables.append(q_table)
    #print(endtime)
    #lines = ("Updating QL_sched with thresh = ", str(thresh)," is = ", str(endtime))
    #with open('sched_updating_time.txt', 'w') as f:
    #    for line in lines:
    #        f.write(line)
    #    f.write('\n')


# In[11]:


lines = ("Updating CLass-based_sched time = ", str(cb_time))

with open('sched_updating_time.txt', 'w') as f:
    for line in lines:
        f.write(line)
    f.write('\n')
    for i in range(len(thresh_list)):
        lines = ("Updating QL_sched with thresh = ", str(thresh_list[i])," is = ", str(upd_times[i]),'\n')
        for line in lines:
            f.write(line)


# In[ ]:




