#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 12:34:26 2023

@author: maq
"""
from random import sample
import numpy as np



reward_transitions = {0:{0:[0,-1,1], 1:[0,-1,1]},
                     1:{0:[0, 0, 0], 1:[0, 0, 0]},
                     2:{0:[0, 0, 0], 1:[0, 0, 0]}} 


state_transitions = {0:{0:[0.1,0.8,0.1], 1:[0.1,0.0,0.9]},
                     1:{0:[0, 0, 0], 1:[0, 0, 0]},
                     2:{0:[0, 0, 0], 1:[0, 0, 0]}} 

actions = [0,1]
states = [0,1,2]
values = [0, 0, 0]

policy = [sample(actions,1)[0] for _ in range(len(states))]
policy = [1, 0, 1]
q_values = [[0 for _ in range(len(actions))] for _ in states]


for _ in range(10):
    values_current = list(values)
    policy_old = list(policy)
    for state in states:
        values_current[state]= np.dot(state_transitions[state][policy[state]],\
                                     np.add(reward_transitions[state][policy[state]], values) )
            
        values = values_current
        
        for action in actions:
            q_values[state][action] = np.dot(state_transitions[state][action],\
                                         np.add(reward_transitions[state][action], values) )
    
    policy = np.argmax(q_values, axis=1)
    if list(policy) == policy_old: 
        break
        
        
        
values = [max(q_s) for q_s in q_values ]
