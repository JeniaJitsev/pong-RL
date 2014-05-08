import random
import os
import pickle

import numpy as np

import nengo
from nengo.utils.distributions import Uniform

from pong_environment_play import PongGame
import rl_pongplayer

class DecoderPongPlayer(nengo.Network):
    def __init__(self, player, env, debug=False, opposite=False, noise=0.0):
        self.state = (0, 0)
        self.debug = debug
        max_y = 480

        def output_func(t, x):
            s, _ = env.move(x[0], player)
#            print "env_state", s
            self.state = [2.0 * x / max_y - 1 for x in s]
            if noise is not None:
                self.state[0] += np.random.normal(scale=noise)

        def direction_func(x):
            if x[0] < x[1]:
                return [1] if opposite else [-1]
            if x[0] > x[1]:
                return [-1] if opposite else [1]
            return [0]

        def state_func(t):
            if self.debug:
                print "state", self.state
            return self.state

        input_node = nengo.Node(state_func)
        e = nengo.Ensemble(nengo.LIF(100), 2,
                           intercepts=Uniform(0, 1),
                           radius=np.sqrt(2))
        output_node = nengo.Node(output_func, size_in=1)

        nengo.Connection(input_node, e)
        nengo.Connection(e, output_node, function=direction_func)

class LearningPongPlayer(nengo.Network):
    def __init__(self, player, env, debug=False):
        self.state = (0, 0)
        self.debug = debug
        max_y = 480

        def output_func(t, x):
            s, self.reward = env.move(x[0], player)
#            print "env_state", s
            self.state = [2.0 * x / max_y - 1 for x in s]

        def direction_func(t, x):
            if x[0] < x[1]:
                return [-1]
            if x[0] > x[1]:
                return [1]
            return [0]
#            return x[0] - x[1]

        def state_func(t):
            if self.debug:
                print "state", self.state
            return self.state

        input_node = nengo.Node(state_func)
        output_node = nengo.Node(output_func, size_in=1)

        ens = nengo.Ensemble(nengo.LIF(100), 2)
        error_ens = nengo.Ensemble(nengo.LIF(100), 1)

        # create learned connection
        nengo.Connection(input_node, ens)

        # relay output so we can send it two different places
        relay = nengo.Node(size_in=1)
        nengo.Connection(ens, relay, function=lambda x: [0],
                         learning_rule=nengo.PES(error_ens, learning_rate=5e-6))
        nengo.Connection(relay, output_node)

        # calculate desired control signal
        error = nengo.Node(direction_func, size_in=2)
        nengo.Connection(input_node, error)
        nengo.Connection(error, error_ens)

        # compare to actual control signal
        nengo.Connection(relay, error_ens, transform= -1)

class decoder_setter:
    def __init__(self, decoder_file):
        with open(decoder_file) as f:
            dt = 0.001
            synapse = 0.01
            self.decoders = np.asarray([[float(x) / (1 - np.exp(-dt / synapse)) for x in line.split()]
                                        for line in f.readlines()])
            # note: dividing by filter to undo the filter that will be applied
            # when the decoders are set

    def __call__(self, A, Y, rng=None):
        return self.decoders, []

SEED = 1
rng = random.Random()
rng.seed(SEED)

nengo.log()

pong_game = PongGame(["computer", "computer"], seed=SEED)

l_rate = 5e-5
discount = 0.97

#d_file = os.path.join("data", "decoders_%s_%s_%s.txt" % (l_rate, discount, SEED))
d_file = os.path.join("data", "decoders_best.txt")
d_solve = decoder_setter(d_file)
#d_solve = nengo.decoders.lstsq_L2nz

with nengo.Network() as net:
    p0 = DecoderPongPlayer(0, pong_game, opposite=True, noise=None)
    p1 = rl_pongplayer.RLPongPlayer(1, pong_game, decoder_solver=d_solve,
                                    l_rate=l_rate, discount=discount, rng=rng)

pong_game.start()
print "Pong started"

sim = nengo.Simulator(net, seed=SEED)
sim.run(2000)
print "done"

# save decoders
#with open(d_file, "w") as f:
#    f.write("\n".join([" ".join([str(x) for x in row]) for row in
#                                sim.signals[sim.model.sig_decoder[p1.learn_conn]].T]))

# save parameters and stats
data = {"l_rate":l_rate,
        "discount":discount,
        "seed":SEED,
        "threshold":p1.threshold,
        "place_dev":p1.place_dev,
        "sample_period":p1.period,
        "stats":p1.stats}

fname = os.path.join("data", "data_%s_%s_%s_%s.pkl" % (l_rate, discount, p1.threshold, SEED))
with open(fname, "wb") as f:
    pickle.dump(data, f)


import matplotlib.pyplot as plt

# plot values over time
plt.figure()
plt.plot(sim.trange(p1.vals_p.sample_every), sim.data[p1.vals_p])
plt.legend(["up", "stay", "down"], loc="lower left")
plt.title("vals")
plt.xlabel("time")
plt.ylabel("action value")

# plot error over time
plt.figure()
plt.plot(sim.trange(p1.err_p.sample_every), sim.data[p1.err_p])
plt.legend(["up", "stay", "down"], loc="lower left")
plt.title("error")
plt.xlabel("time")
plt.ylabel("error")

#plt.figure()
#plt.plot(sim.trange(p1.err_p.sample_every), sim.data[p1.err_p][:, 1])
#plt.legend(["up", "stay", "down"], loc="lower left")

# plot avg decoder by encoder direction
plt.figure()
enc = sim.model.params[p1.vals].encoders
dec = sim.signals[sim.model.sig_decoder[p1.learn_conn]].T
colours = [-1, 0, 1]
colour_array = [np.dot(colours, d) for d in dec]
#plt.quiver(np.zeros(1000), np.zeros(1000), enc[:, 0], enc[:, 1],
#           colour_array, scale=2.5)
plt.scatter([p1.placecells[np.argmax(e)][0] + rng.random() * 0.1 for e in enc],
            [p1.placecells[np.argmax(e)][1] + rng.random() * 0.1 for e in enc],
            c=colour_array,
            vmin=np.mean(colour_array) * 0.25, vmax=np.mean(colour_array) * 1.75)
plt.title("action vals arranged by encoder")
plt.xlabel("ball")
plt.ylabel("paddle")

# plot max val by encoder direction
plt.figure()
#colour_array = [max(d) for d in dec]
#plt.scatter([p1.placecells[np.argmax(e)][0] + rng.random() * 0.1 for e in enc],
#            [p1.placecells[np.argmax(e)][1] + rng.random() * 0.1 for e in enc],
#            c=colour_array, vmin=np.mean(colour_array) * 0.25, vmax=np.mean(colour_array) * 1.75)
Z = np.zeros((100, 100))
radius = p1.state_radius
indices = (100 / (2 * radius) * (sim.data[p1.state_p][:, :2] + radius)).astype(int)
Z[indices[:, 0], indices[:, 1]] = np.max(sim.data[p1.vals_p], axis=1)

plt.contour(np.linspace(-radius, radius, 100),
            np.linspace(-radius, radius, 100),
            Z)
plt.title("max vals arranged by location")
plt.xlabel("paddle")
plt.ylabel("ball")

# plot values when state is within some range
plt.figure()
state = sim.data[p1.state_p]
#print np.shape(state)
#print min(state[:, 0]), max(state[:, 1])
#pts = (state[:, 0] < -0.2) & (state[:, 0] > -0.5) & \
#    (state[:, 1] > 0.2) & (state[:, 1] < 0.5)
pts = state[:, 0] < state[:, 1]
data = sim.data[p1.vals_p].copy()
data[~pts] = np.nan
plt.plot(sim.trange(p1.vals_p.sample_every),
         data)
plt.xlim([0, max(sim.trange())])
plt.legend(["up", "stay", "down"], loc="lower left")
plt.title("vals when ball above paddle")
plt.xlabel("time")
plt.ylabel("action value")

# plot state
plt.figure()
plt.plot(sim.trange(p1.state_p.sample_every), sim.data[p1.state_p][:, :2])
plt.legend(["ball", "paddle"])
plt.title("state")
plt.xlabel("time")
plt.ylabel("position")

# plot state spikes
#from nengo.utils.matplotlib import rasterplot
#plt.figure()
#rasterplot(sim.trange(p1.val_spikes.sample_every), sim.data[p1.val_spikes])
#plt.plot(sim.trange(p1.val_spikes.sample_every), sim.data[p1.val_spikes])

# plot hits/misses
plt.figure()
#print np.shape(p1.stats), np.shape(sim.trange(0.25))
plt.plot(sim.trange(p1.period), p1.stats[2:])
plt.legend(["p1_hit", "p2_hit", "p1_miss", "p2_miss"], loc="upper left")
plt.title("stats")
plt.xlabel("time")
plt.ylabel("count")

# plot sparsity
plt.figure()
spikes = sim.data[p1.val_spikes]
plt.plot(sim.trange(p1.val_spikes.sample_every),
         [np.count_nonzero(s) / float(len(s)) for s in spikes])
plt.title("sparsity")
plt.xlabel("time")
plt.ylabel("neurons active")

# plot values normalized to 0
plt.figure()
vals = sim.data[p1.vals_p]
mins = np.min(vals, axis=1)
plt.plot(sim.trange(p1.vals_p.sample_every), vals - np.tile(mins, (3, 1)).T)
plt.legend(["up", "stay", "down"], loc="upper left")
plt.title("normalized values")
plt.xlabel("time")
plt.ylabel("action value")


plt.show()



