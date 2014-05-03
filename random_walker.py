import random
from pong_environment_play import PongGame
import time

env = PongGame(["human", "computer"])
env.start()
while True:
	time.sleep(0.05)
	direction = random.randint(-1, 1)

	outcome = 0
	state, outcome = env.move(direction, 1)



