import nengo
import random
import math
#import pong_environment_play as env
from pong_environment_play import PongGame
import time

class PongPlayer(nengo.Network):
    def __init__(self, player, env, debug=False):
        self.state = (0, 0)
        self.debug = debug
        max_y = 480

        def output_func(t, x):
            s, _ = env.move(x[0], player)
#            print "env_state", s
            self.state = [2.0 * x / max_y - 1 for x in s]

        def direction_func(x):
            if x[0] < x[1]:
                return [-1]
            if x[0] > x[1]:
                return [1]
            return [0]

        def state_func(t):
            if self.debug:
                print "state", self.state
            return self.state

        input_node = nengo.Node(state_func)
        e = nengo.Ensemble(nengo.LIF(100), 2)
        output_node = nengo.Node(output_func, size_in=1)

        nengo.Connection(input_node, e)
        nengo.Connection(e, output_node, function=direction_func)

pong_game = PongGame(["computer", "computer"])

with nengo.Network() as net:
    p0 = PongPlayer(0, pong_game)
    p1 = PongPlayer(1, pong_game, debug=False)


pong_game.start()
print "Pong started"

sim = nengo.Simulator(net)
sim.run(10000)
print "done"


