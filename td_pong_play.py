import random
import numpy
import sys
import time
import os
import json
from pong_environment_play import PongGame

policy_filename = "pong_policy.dat"
values_filename = "pong_values.dat"

alpha = 0.1 # values / critic learning parameter
beta = 0.1  # actor learning parameter
gamma = 0.9  # error signal: future states parameter

env = PongGame(["human", "computer"], bins=12)

world_dim = env.getWorldDim()
num_possible_moves = env.getActionDim()

state = env.getState(1)

pol_file = None
val_file = None

if os.path.exists(policy_filename):
    pol_file = open(policy_filename, 'r+')
    policy = numpy.array(json.loads(pol_file.read()))
    pol_file.close()
else:
    #create random policy
    policy = numpy.random.rand(world_dim[1], world_dim[0], num_possible_moves)

#pol_file = open(policy_filename, 'w+')

if os.path.exists(values_filename):
    val_file = open(values_filename, 'r+')
    values = numpy.array(json.loads(val_file.read()))
    val_file.close()
else:
    #create empty value funcion
    values = numpy.zeros([world_dim[1], world_dim[0]])



#val_file = open(values_filename, 'w+')

def cum_softmax_direction_prop(state):
    # calculates the cumulated softmax propability for every possible action
    current_policy = policy[state[1], state[0], :]  # prop in this agent_pos
    softmax_prop = numpy.exp(current_policy)
    softmax_prop = softmax_prop / numpy.sum(softmax_prop)  # softmax: (e^prop) / (sum(e^prop))
    cum_softmax_prop = numpy.cumsum(softmax_prop)  # cumulating
    return (cum_softmax_prop)


def pick_action(state):
    cum_softmax_prop = cum_softmax_direction_prop(state)
    r = numpy.random.rand()
    for i in range(len(cum_softmax_prop)):
        if cum_softmax_prop[i] > r:
            return i


def critic(state, last_state, reward):
    error = reward - values[last_state[1], last_state[0]] + gamma * values[state[1], state[0]]
    return (error)


env.start()
while True:

    direction = pick_action(state)

    #	last_state = state[:][:] # NO COPY??


    outcome = 0
    last_state, outcome = env.move(direction, 1)

    time.sleep(0.05)
    state = env.getState(1)

    if outcome != 0 or state != last_state:
        error = critic(state, last_state, outcome)
        #	print "error ", error
        values[last_state[1], last_state[0]] += alpha * error

        policy[last_state[1], last_state[0], direction] += beta * error

    #	if outcome != 0:
    #		for row in values:
    #			print numpy.array(row, dtype=int) 


#	pol_file.seek(0)
#	val_file.seek(0)
#	pol_file.write(json.dumps(policy.tolist()))
#	val_file.write(json.dumps(values.tolist()))
#	pol_file.truncate()
#	val_file.truncate()
#
#
#pol_file.close()
#val_file.close()
