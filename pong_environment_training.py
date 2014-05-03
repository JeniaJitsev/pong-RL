import time
import numpy.random as rand
import sys

world_dim = {'ball_y':12, 'paddle': 12}
num_possible_moves = 3


global num_reward, num_punishment, state
num_reward = 1
num_punishment = 1

state = (1,0) # ball_x, ball_y, paddle
last_outcomes = [0]

beam_next = False
beam_timer = 2

def getWorldDim():
	return [world_dim['ball_y'], world_dim['paddle']] 

def getActionDim():
	return num_possible_moves

def checkValid():
	global state
	if state[1] < 0:
		state = (state[0], 0) # -= 1
	if state[1] > world_dim['paddle']-1:
		state = (state[0], world_dim['paddle']-1) # -= 1

def randomBeam():
	global state
	state = ( rand.randint(0, world_dim['ball_y']), rand.randint(0, world_dim['paddle']) )

def move(direction):
	global num_reward, num_punishment, state, beam_next, beam_timer

	if direction == 2:
		direction = -1 #convert to pong actions 


	if direction == -1:
		state = (state[0], state[1]-1) # -= 1
	elif direction == 1:
		state = (state[0], state[1]+1)
	
	if beam_next:
		if beam_timer <= 0:
			randomBeam()
			beam_next = False
			beam_timer = 2
		else:
			beam_timer -= 1
	
	checkValid()	

	outcome = -10
	if (state[0] == state[1]):
		outcome = 600
		beam_next = True
	
	return [state, outcome]


def getState():
	return state



