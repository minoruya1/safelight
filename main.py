#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division

import glob
import logging
import shutil
import subprocess
import time

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
# import matplotlib.pyplot as plt
import scipy.misc
import os
import scipy.stats as ss
import math
import numpy
import pandas as pd

# matplotlib inline


# In[ ]:


# The class of dueling DQN with three convolutional layers
class Qnetwork():
    def __init__(self, h_size, action_num):
        # The network recieves a state from the sumo, flattened into an array.
        # It then resizes it and processes it through three convolutional layers.
        self.scalarInput = tf.placeholder(shape=[None, 3600, 3], dtype=tf.float32)
        self.domainActionInput = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.legal_actions = tf.placeholder(shape=[None, action_num], dtype=tf.float32)

        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 60, 60, 3])
        # with slim.arg_scope([slim.conv2d, slim.fully_connected],
        #                     activation_fn=tf.nn.relu,
        #                     weights_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
        #                     weights_regularizer=slim.l2_regularizer(0.0005)):
        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=32, kernel_size=[4, 4], stride=[2, 2],
                                 padding='VALID', activation_fn=self.relu, biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[2, 2], stride=[1, 1],
                                 padding='VALID',
                                 activation_fn=self.relu, biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=128, kernel_size=[2, 2], stride=[1, 1],
                                 padding='VALID',
                                 activation_fn=self.relu, biases_initializer=None)

        # It is split into Value and Advantage
        self.stream = slim.flatten(self.conv3)
        self.stream0 = slim.fully_connected(self.stream, 128, activation_fn=self.relu)
        self.stream1 = slim.fully_connected(self.domainActionInput, 128, activation_fn=self.relu)
        self.stream0 = self.stream1 + self.stream0

        self.streamA = self.stream0
        self.streamV = self.stream0

        self.streamA0 = slim.fully_connected(self.streamA, h_size, activation_fn=self.relu)
        self.streamV0 = slim.fully_connected(self.streamV, h_size, activation_fn=self.relu)

        xavier_init = tf.contrib.layers.xavier_initializer()
        action_num = np.int32(action_num)
        self.AW = tf.Variable(xavier_init([h_size, action_num]))
        self.VW = tf.Variable(xavier_init([h_size, 1]))
        self.Advantage = tf.matmul(self.streamA0, self.AW)
        self.Advantage = tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))\
                         /(tf.math.reduce_std(self.Advantage, axis=1, keepdims=True) + 1e-5)
        self.Value = tf.matmul(self.streamV0, self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout0 = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        # The final Q value is the addition of the Q value and penelized value for illegal actions
        self.Qout = tf.add(self.Qout0, self.legal_actions)
        self.Q_softmax = tf.math.softmax(self.Qout0)
        # The predicted action
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the mean square error between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, np.int32(action_num), dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        #         self.Q = tf.reduce_sum(self.Qout, axis=1)

        self.safe_q = tf.placeholder(shape=[None, action_num], dtype=tf.float32)
        self.action_q = tf.placeholder(shape=[None, action_num], dtype=tf.float32)
        self.td_error = tf.square(self.targetQ - self.Q)
        k = tf.keras.losses.KLDivergence(reduction=tf.compat.v1.losses.Reduction.NONE)
        self.safe_penalty = tf.square(k(self.Advantage, self.safe_q))
        self.safe_penalty_scaler = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.safe_penalty = tf.multiply(self.safe_penalty, tf.transpose(self.domainActionInput))
        # self.safe_penalty = tf.reduce_mean(self.action_q - self.safe_q)
        # self.safe_penalty = tf.reduce_mean(tf.square(self.safe_q - self.Q_softmax))
        self.loss = tf.reduce_mean(self.td_error) + tf.reduce_sum(self.safe_penalty) * 1e-4 * tf.reduce_sum(self.safe_penalty_scaler)
        # self.loss = tf.reduce_sum(self.safe_penalty) * tf.reduce_sum(self.safe_penalty_scaler)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

    def relu(self, x, alpha=0.01, max_value=None):
        '''ReLU.

        alpha: slope of negative section.
        '''
        negative_part = tf.nn.relu(-x)
        x = tf.nn.relu(x)
        if max_value is not None:
            x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                                 tf.cast(max_value, dtype=tf.float32))
        x -= tf.constant(alpha, dtype=tf.float32) * negative_part
        return x


# In[ ]:


# The normal experience buffer
class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)
        print
        "ADDED", len(self.buffer)

    def sample(self, size):
        print
        "BUFFer:", len(self.buffer)
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 6])


# In[ ]:


# The target update functions
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


# In[ ]:


# The parameters
batch_size = 128  # How many experiences to use for each training step.
update_freq = 4  # How often to perform a training step.
y = .99  # Discount factor on the target Q-values
startE = 1  # Starting chance of random action
endE = 0.01  # Final chance of random action
anneling_steps = 10000.  # How many steps of training to reduce startE to endE.
num_episodes = 800  # 000 #How many episodes of game environment to train network with.
pre_train_steps = 2000  # 0000 #How many steps of random actions before training begins.
max_epLength = 500  # The max allowed length of our episode.
load_model = False  # Whether to load a saved model.
action_num = 13  # total number of actions
path = "./dqn"  # The path to save our model to.
h_size = 64  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001  # Rate to update target network toward primary network

# In[ ]:


import math


class priorized_experience_buffer():
    def __init__(self, buffer_size=20000):
        self.buffer = []
        self.prob = []
        self.err = []
        self.buffer_size = buffer_size
        self.alpha = 0.2

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.err[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
            self.prob[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)
        self.err.extend([10000] * len(experience))
        self.prob.extend([1] * len(experience))

    def updateErr(self, indx, error):
        for i in range(0, len(indx)):
            self.err[indx[i]] = math.sqrt(error[i])
        r_err = ss.rankdata(self.err)  # rank of the error from smallest (1) to largest
        self.prob = [1 / (len(r_err) - i + 1) for i in r_err]

    def priorized_sample(self, size):
        prb = [i ** self.alpha for i in self.prob]
        t_s = [prb[0]]
        for i in range(1, len(self.prob)):
            t_s.append(prb[i] + t_s[i - 1])
        batch = []
        mx_p = t_s[-1]

        smp_set = set()

        while len(smp_set) < batch_size:
            tmp = np.random.uniform(0, mx_p)
            for j in range(0, len(t_s)):
                if t_s[j] > tmp:
                    smp_set.add(max(j - 1, 0))
                    break;
        for i in smp_set:
            batch.append([self.buffer[i], i])
        return np.array(batch)

    # smp_set = np.random.choice(np.arange(len(self.prob)), batch_size,
    #                            p=np.array(self.prob) / np.sum(np.array(self.prob)))


#         return np.reshape(np.array(random.sample(self.buffer,size)),[size,6])


# In[ ]:


# The code for the SUMO environment
import os, sys

# if 'SUMO_HOME' in os.environ:

# The path of SUMO-tools to get the traci library
sys.path.append(os.path.join('/home/ring/sumo-svn/', 'tools'))
from sumolib import checkBinary
import traci
import traci.constants as tc
import xml.etree.cElementTree as ET
import numpy as np

import datetime
# Environment Model
DEFAULT_PORT = 8813
# if gui:
#     app = 'sumo-gui'
# else:
#     app = 'sumo'
# sumoBinary = "/usr/local/bin/sumo"
sumocfg_file = "cross/cross1.sumocfg"
# sumoCmd = [sumoBinary, "-c", "./cross/cross.sumo.cfg"]  # The path to the sumo.cfg file
# command += ['--remote-port', str(DEFAULT_PORT)]


# reset the environment
def reset(is_analysis=False, sumocfg_file = sumocfg_file):
    if is_analysis:
        sumocfg_file= "cross/cross_wooutput.sumocfg"
    command = [checkBinary('sumo'), '-c', sumocfg_file]
    # traci.start(sumoCmd)
    # subprocess.Popen(command)
    # wait 2s to establish the traci server
    # time.sleep(2)
    # sim = traci.connect(port= DEFAULT_PORT)

    #     command += ['--remote-port', '8813' ]
    #     command += ['--collision-output ', 'collision.xml']
    #     command += ['--statistic-output', 'statistic.xml']
    traci.start(command)
    tls = traci.trafficlight.getIDList()
    traci.trafficlight.setProgram(tls[0],'0')
    return tls


# get the starting state
def state():
    for veh_id in traci.vehicle.getIDList():
        traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED))
    # if len(traci.vehicle.getIDList()) != 0:
    p = traci.vehicle.getAllContextSubscriptionResults()
    # else:
    #     p = []
    p_state = np.zeros((60, 60, 3))
    for x in p:
        ps = p[x][tc.VAR_POSITION]
        spd = p[x][tc.VAR_SPEED]
        ttc = traci.vehicle.getParameter(x, "device.ssm.minTTC")
        ttc = 0 if ttc == '' else ttc
        p_state[int(ps[0] / 5), int(ps[1] / 5)] = [1, int(round(spd)), round(float(ttc))]
    #         v_state[int(ps[0]/5), int(ps[1]/5)] = spd
    p_state = np.reshape(p_state, [-1, 3600, 3])
    return p_state  # , v_state]


# get the legal actions at the current phases of the traffic light
def getLegalAction(phases):
    legal_action = np.zeros(13, dtype=numpy.int8) - 1
    i = 0
    for x in phases:
        if x > 5:
            legal_action[i] = i
            if i == 0:
                legal_action[9] = 9
            if i == 2:
                legal_action[10] = 10
        if x < 60:
            legal_action[i + 5] = i + 5
            if i == 0:
                legal_action[11] = 11
            if i == 2:
                legal_action[12] = 12
        i += 1
    legal_action[4] = 4
    return legal_action


# get the new phases after taking action from the current phases
def getPhaseFromAction(phases, act):
    if act < 4:
        phases[int(act)] -= 5
    elif 4 < act < 9:
        phases[int(act) - 5] += 5
    elif act == 9:
        phases[0] -= 5
    elif act == 10:
        phases[2] -= 5
    elif act == 11:
        phases[0] += 5
    elif act == 12:
        phases[2] += 5
    return phases

collision_flag = -1
# the process of the action
# input: traffic light; new phases; waiting time in the beginning of this cycle
# output: new state; reward; End or not(Bool); new waiting time at the end of the next cycle
def action(tls, ph, act, wait_time):  # parameters: the phase duration in the green signals
    global collision_flag
    tls_id = tls[0]
    init_p = traci.trafficlight.getPhase(tls_id)
    prev = -1
    changed = False
    current_phases = ph
    collision_phase = -1
    collsions = 0
    p_state = np.zeros((60, 60, 3))
    if act < 9:
        traci.trafficlight.setProgram(tls_id, '0')
    if act == 9 or act == 11:
        traci.trafficlight.setProgram(tls_id, '1')
    elif act == 10 or act == 12:
        traci.trafficlight.setProgram(tls_id, '2')
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        c_p = traci.trafficlight.getPhase(tls_id)
        # if c_p == 0 and (act == 9 or act == 11):
        #     traci.trafficlight.setRedYellowGreenState(tls_id,'GGGrrrrrGGGrrrrr')
        if c_p != prev and c_p % 2 == 0:
            traci.trafficlight.setPhaseDuration(tls_id, ph[int(c_p / 2)] - 0.5)
            prev = c_p
        if init_p != c_p:
            changed = True
        if changed:
            if c_p == init_p:
                break
        traci.simulationStep()
        # collision_flag=collect_collision_info(collision_flag)
        collision_t = traci.simulation.getCollidingVehiclesNumber()
        if collision_t > 0:
            collision_phase = c_p
            if c_p != 0 and c_p != 1 and c_p != 4 and c_p != 5:
                print("------collision occur when signal phase is %d" % c_p)
            collsions += collision_t
        step += 1
        if step % 10 == 0:
            for veh_id in traci.vehicle.getIDList():
                wait_time_map[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
    for veh_id in traci.vehicle.getIDList():
        traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_ACCUMULATED_WAITING_TIME))
    p = traci.vehicle.getAllSubscriptionResults()

    wait_temp = dict(wait_time_map)
    for x in p:
        ps = p[x][tc.VAR_POSITION]
        spd = p[x][tc.VAR_SPEED]
        ttc = traci.vehicle.getParameter(x,"device.ssm.minTTC")
        ttc = 0 if ttc == '' else ttc
        p_state[int(ps[0] / 5), int(ps[1] / 5)] = [1, int(round(spd)), round(float(ttc))]

    wait_t = sum(wait_temp[x] for x in wait_temp)

    d = False
    if traci.simulation.getMinExpectedNumber() == 0:
        d = True

    r = wait_time - wait_t
    p_state = np.reshape(p_state, [-1, 3600, 3])
    return p_state, r, d, wait_t, collision_phase, collsions


# close the environment after every episode
def end():
    traci.close()


def get_safe_action(collision_ph, default_ph, domain_action):
    if collision_ph == 0 or collision_ph == 1 or domain_action[0][0] == 9 or domain_action[0][0] == 11:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif collision_ph == 4 or collision_ph == 5 or domain_action[0][0] == 10 or domain_action[0][0] == 12:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    else:
        return default_ph


def collect_collision_info(collision_flag=-1):
    current_step = int(traci.simulation.getTime() * 2)
    traci.simulation.saveState("state_%s.xml" % current_step)
    if current_step > 5:
        cmd = 'del ' + "state_%s.xml" % (current_step - 5)
        subprocess.check_call(cmd, shell=True)
    if collision_flag != -1 and current_step-collision_flag <= 5:
        traci.simulation.saveState("./collision_%s/state_%s.xml" % (collision_flag,current_step))
    else:
        collision_flag = -1
    if traci.simulation.getCollidingVehiclesNumber() > 0:
        os.mkdir("../singleIntersection/collision_%s" % current_step)
        for file in glob.glob('../singleIntersection/state*.xml'):
            shutil.copy(file,  '../singleIntersection/collision_%s/' % current_step)
        collision_flag = current_step
    return collision_flag


def remove_trash_collisions():
    for file in glob.glob('../singleIntersection/state*.xml'):
        try:
            os.remove(file)
        except:
            print("Error while deleting file : ", file)
    for directory in glob.glob('../singleIntersection/collision_*'):
        shutil.rmtree(directory)

    #     print(traci.simulation.getCollisions())
    #     for vehicle in traci.simulation.getCollidingVehiclesIDList():
    #         traci.vehicle.highlight(vehicle)
    #     # traci.load(['--breakpoints', str(traci.simulation.getTime())])
    #     traci.simulation.saveState("state_%s.xml" % step)
        # traci.gui.screenshot("View #0", "collision_%s.png" % step)


def parse_collisions():
    reset(is_analysis=True)
    collision_statistics = './cross/collision.xml'
    tree = ET.ElementTree(file=collision_statistics)
    for child in tree.getroot():
        curr_collision = child.attrib
        collision_time = int(float(curr_collision['time']) * 2) + 1
        root_dir = '../singleIntersection/collision_%s/' % str(collision_time)
        for i, file in enumerate(os.listdir(root_dir)):
            traci.simulation.loadState(root_dir + file)

            vehicle_list = traci.vehicle.getIDList()
            if curr_collision['collider'] in vehicle_list:
                # traci.vehicle.highlight(curr_collision['collider'], size=1,alphaMax=5, duration=5)
                traci.vehicle.setColor(curr_collision['collider'],(255, 0, 0, 255))
                # traci.gui.trackVehicle("View #0", curr_collision['collider'])
            if curr_collision['victim'] in vehicle_list:
                traci.vehicle.setColor(curr_collision['victim'], (255, 0, 0, 255))
                # traci.vehicle.highlight(curr_collision['victim'], size=1, alphaMax=5, duration=5)
            time.sleep(1)
            traci.gui.screenshot("View #0", "./collision_%s/snapshot_%s_%s.png" %
                                 (str(collision_time), str(collision_time), str(i)))
            traci.simulation.step()

    end()

left_turn_lanes = ['4i_2', '2i_2', '3i_2', '1i_2']


opposing_lanes = {
    '1i_2': ['2i_0', '2i_1'],
    '2i_2': ['1i_0', '1i_1'],
    '3i_2': ['4i_0', '4i_1'],
    '4i_2': ['3i_0', '3i_1']
}


def domain_model():
    protected_left_turn_lane = []
    modified_left_turn = ''
    for left_turn_lane in left_turn_lanes:
        link_info = traci.lane.getLinks(left_turn_lane)[0]
        if traci.lane.getLastStepOccupancy(left_turn_lane) > 0 and link_info[5] == 'g' and link_info[3]:
            protected_left_turn_lane.append(left_turn_lane)
    if len(protected_left_turn_lane) > 1:
        modified_left_turn = np.random.choice(protected_left_turn_lane, 1)[0]
    if modified_left_turn == '4i_2' or modified_left_turn == '3i_2':
        return np.array([[1]]) if np.random.random(1)[0] < 0.5 else np.array([[1]])
    elif modified_left_turn == '1i_2' or modified_left_turn == '2i_2':
        return np.array([[1]]) if np.random.random(1)[0] < 0.5 else np.array([[1]])
    return np.array([[0]])
# In[ ]:

# parse_collisions()
tf.reset_default_graph()
# define the main QN and target QN
mainQN = Qnetwork(h_size, np.int32(action_num))
targetQN = Qnetwork(h_size, np.int32(action_num))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)

# define the memory
myBuffer0 = priorized_experience_buffer()

# Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE) / anneling_steps

# create lists to contain total rewards and steps per episode
jList = []  # number of steps in one episode
rList = []  # reward in one episode
wList = []  # the total waiting time in one episode
awList = []  # the average waiting time in one episode
tList = []  # thoughput in one episode (number of generated vehicles)
nList = []  # stops' percentage (number of stopped vehicles divided by the total generated vehicles)
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

init_phases = [20, 20, 20, 20]

sess = tf.InteractiveSession()

# record the loss
tf.summary.scalar('Loss', mainQN.loss)

data = []
rfile = open(path + '/reward-rl.csv', 'w')
wfile = open(path + '/wait-rl.csv', 'w')
awfile = open(path + '/acc-wait-rl.csv', 'w')
tfile = open(path + '/throput-rl.csv', 'w')

merged = tf.summary.merge_all()
s_writer = tf.summary.FileWriter(path + '/train', sess.graph)
s_writer.add_graph(sess.graph)

sess.run(init)
tf.global_variables_initializer().run()
if load_model == True:
    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(path)
    saver.restore(sess, ckpt.model_checkpoint_path)
updateTarget(targetOps, sess)  # Set the target network to be equal to the primary network.

# In[ ]:


# the running process of the total number of episodes
for i in range(1, num_episodes):
    # remove_trash_collisions()
    episodeBuffer0 = priorized_experience_buffer()
    # Reset environment and get first new observation
    tls = reset()
    s = state()  # np.random.rand(1,10000)
    domain_action = domain_model()
    current_phases = list(init_phases)
    wait_time_map = {}
    wait_time = 0 #newly added
    d = False
    rAll = 0
    j = 0
    collision_count = 0
    print
    "III:", i, e
    while j < max_epLength:
        j += 1

        # get the legal actions at the current state
        legal_action = getLegalAction(current_phases)  # np.random.randint(1,action_num,size=action_num) #[1,2,-1,4,5]

        # Choose an action (0-8) by greedily (with e chance of random action) from the Q-network
        if np.random.rand(1) < e or total_steps < pre_train_steps:
            a_cnd = [x for x in legal_action if x != -1]
            a_num = len(a_cnd)
            a = np.random.randint(0, a_num)
            a = a_cnd[a]
            legal_a_one = [0 if x != -1 else -99999 for x in legal_action]
            adv = sess.run([mainQN.Advantage],
                     feed_dict={mainQN.scalarInput: s,mainQN.domainActionInput: domain_action, mainQN.legal_actions: [legal_a_one]})
            adv = adv[0]
            # pi = np.zeros(action_num)
            # pi[a] = 1
        else:
            np.reshape(s, [-1, 3600, 3])
            legal_a_one = [0 if x != -1 else -99999 for x in legal_action]
            a_out, adv = sess.run([mainQN.predict, mainQN.Advantage],feed_dict={mainQN.scalarInput: s,
                                                                                mainQN.domainActionInput: domain_action,
                                                                                mainQN.legal_actions: [legal_a_one]})
            a = a_out[0]
            adv = adv[0]

        ph = getPhaseFromAction(current_phases, a)
        s1, r, d, wait_time, collision_ph, collisions = action(tls, ph, a, wait_time)
        current_phases = ph
        collision_count += collisions
        total_steps += 1
        legal_a_one = [0 if x != -1 else -99999 for x in legal_action]  # the penalized Q value for illegal actions
        legal_act_s1 = getLegalAction(ph)
        legal_a_one_s1 = [0 if x != -1 else -99999 for x in legal_act_s1]
        domain_action_1 = domain_model()
        desired_action_distribution = get_safe_action(collision_ph, adv, domain_action)
        episodeBuffer0.add(np.reshape(np.array([s, a, r, s1, d, legal_a_one, legal_a_one_s1, desired_action_distribution, adv, collisions,
                                                domain_action, domain_action_1]),
                                      [1, 12]))  # Save the experience to our episode buffer.

        if total_steps > pre_train_steps:
            if e > endE:
                e -= stepDrop
            if total_steps % (update_freq) == 0:
                trainBatch = myBuffer0.priorized_sample(batch_size)  # Get a random batch of experiences.
                indx = np.reshape(np.vstack(trainBatch[:, 1]), [batch_size])
                indx = indx.astype(int)
                trainBatch = np.vstack(trainBatch[:, 0])

                # Below we perform the Double-DQN update to the target Q-values
                # action from the main QN
                Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3]),
                                                         mainQN.domainActionInput: np.vstack(trainBatch[:, 11]),
                                                         mainQN.legal_actions: np.vstack(trainBatch[:, 6])})
                # Q value from the target QN
                Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3]),
                                                        targetQN.domainActionInput: np.vstack(trainBatch[:, 11]),
                                                        targetQN.legal_actions: np.vstack(trainBatch[:, 6])})
                # get targetQ at s'
                end_multiplier = -(trainBatch[:, 4] - 1)  # if end, 0; otherwise 1
                doubleQ = Q2[range(batch_size), Q1]
                targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)

                # Update the network with our target values.
                summry, err, ls, md, kl, advantage, safe_q = sess.run([merged, mainQN.td_error, mainQN.loss, mainQN.updateModel, mainQN.safe_penalty, mainQN.Advantage, mainQN.safe_q],
                                               feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]),
                                                          mainQN.domainActionInput: np.vstack(trainBatch[:, 10]),
                                                          mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:, 1],
                                                          mainQN.legal_actions: np.vstack(trainBatch[:, 5]),
                                                          mainQN.safe_q: np.vstack(trainBatch[:, 7]),
                                                          mainQN.action_q: np.vstack(trainBatch[:, 8]),
                                                          mainQN.safe_penalty_scaler: np.vstack(trainBatch[:, 9])})

                s_writer.add_summary(summry, total_steps)
                # update the target QN and the memory's prioritization
                updateTarget(targetOps, sess)  # Set the target network to be equal to the primary network.
                myBuffer0.updateErr(indx, err)

        rAll += r
        s = s1
        domain_action = domain_action_1

        if d == True:
            break
    end()
    # parse_collisions()
    # save the data into the myBuffer
    myBuffer0.add(episodeBuffer0.buffer)

    jList.append(j)
    rList.append(rAll)
    rfile.write(str(rAll) + '\n')
    wt = sum(wait_time_map[x] for x in wait_time_map)
    wtAve = wt / len(wait_time_map)
    wList.append(wtAve)
    wfile.write(str(wtAve) + '\n')
    awList.append(wt)
    awfile.write(str(wt) + '\n')
    tList.append(len(wait_time_map))
    tfile.write(str(len(wait_time_map)) + '\n')
    tmp = [x for x in wait_time_map if wait_time_map[x] > 1]
    nList.append(len(tmp) / len(wait_time_map))

    log = {'collisions': collision_count/2,
           'step': total_steps,
           'episode_steps': j,
           'reward': rAll,
           'wait_average': wtAve,
           'accumulated_wait': wt,
           'throughput': len(wait_time_map),
           'stops': len(tmp) / len(wait_time_map)}
    data.append(log)
    # self._add_summary(mean_reward, global_step)
    # self.summary_writer.flush()
    logging.info('''Training: episode %d, total_steps: %d, sum_reward: %.2f, collisions: %d''' %
                 (i, total_steps, rAll, collision_count/2))
    df = pd.DataFrame(data)
    df.to_csv('train_log.csv')

    print
    "Total Reward---------------", rAll
    # Periodically save the model.
    if i % 100 == 0:
        saver.save(sess, path + '/model-' + str(i) + '.cptk')
        print("Saved Model")
#         if len(rList) % 10 == 0:
#             print(total_steps,np.mean(rList[-10:]), e)
saver.save(sess, path + '/model-' + str(i) + '.cptk')
print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")

# In[ ]:
import pandas as pd
import matplotlib.pyplot as plt
sample_df = pd.read_csv ('train_log.csv')
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
sample_df["mean"] = sample_df["reward"].rolling(10).mean()
sample_df["std"] = sample_df["reward"].rolling(10).std()
ax.fill_between(sample_df.index,
                sample_df["mean"]-sample_df["std"],
                sample_df["mean"]+sample_df["std"],
                alpha=0.2)
ax.plot(sample_df.index,sample_df["mean"], '-', label="reward")

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("training_reward.png")

# save the data again in case data are missed in the previous step
rfile = open(path + '/reward-rl.csv', 'w')
wfile = open(path + '/wait-rl.csv', 'w')
awfile = open(path + '/acc-wait-rl.csv', 'w')
tfile = open(path + '/throput-rl.csv', 'w')
jfile = open(path + '/epi-len-rl.csv', 'w')
nfile = open(path + '/stop-rl.csv', 'w')

for x in rList:
    rfile.write(str(x) + '\n')
for x in wList:
    wfile.write(str(x) + '\n')
for x in awList:
    awfile.write(str(x) + '\n')
for x in tList:
    tfile.write(str(x) + '\n')
for x in jList:
    jfile.write(str(x) + '\n')
for x in nList:
    nfile.write(str(x) + '\n')

import matplotlib.pyplot as plt

x = range(1, len(rList) + 1)

plt.scatter(x, rList)
plt.show()
plt.savefig(path + "/reward.png")

plt.scatter(x, wList)
plt.show()
plt.savefig(path + "/wait.png")

plt.scatter(x, awList)
plt.show()
plt.savefig(path + "/acc-wait.png")

plt.scatter(x, jList)
plt.show()
plt.savefig(path + "/epi-len.png")

plt.scatter(x, nList)
plt.show()

# In[ ]:




