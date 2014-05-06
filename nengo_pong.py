import nengo
from nengo.utils.distributions import Uniform
import numpy as np
import copy
import random
from pong_environment_play import PongGame

class DecoderPongPlayer(nengo.Network):
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
        e = nengo.Ensemble(nengo.LIF(100), 2,
                           intercepts=Uniform(0, 1))
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

class TDErrorCalc:
    def __init__(self, period, num_actions, discount):
        self.period = period
        self.saved_vals = [0 for _ in range(num_actions)]
        self.reward = 0
        self.discount = discount

    def __call__(self, t, x):
        self.reward += x[0]

        if t % self.period <= 0.001:
#            print t

            vals = x[1:]

            err = self.reward + self.discount * max(vals) - max(self.saved_vals)
            self.err_sig = [err if i == np.argmax(self.saved_vals) else
                            0 for i in range(len(self.saved_vals))]

            print "%.2e %.2e %.2e %.2e" % (err, self.reward, max(vals), max(self.saved_vals))

            self.reward = 0
            self.saved_vals = copy.copy(vals)

#            print "err %.2f %s" % (err, self.err_sig)
        return self.err_sig

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

class RLPongPlayer(nengo.Network):
    def __init__(self, player, env, decoder_solver=None, l_rate=1e-7):
        num_actions = 3
        self.state = (0, 0)
        max_y = 480
        period = 0.25
        reward_scale = 0.0001

        def output_func(t, x):
            s, self.reward = env.move(x[0], player)
            self.reward *= reward_scale
            self.state = [2.0 * x / max_y - 1 for x in s]

        def state_func(t):
            return self.state

        input_node = nengo.Node(state_func)
        output_node = nengo.Node(output_func, size_in=1)

        td_node = nengo.Node(TDErrorCalc(period, num_actions, 0.9),
                             size_in=num_actions + 1, label="td_node")
        td_relay = nengo.Node(size_in=num_actions)
        nengo.Connection(td_node, td_relay)

        reward_node = nengo.Node(lambda t: [self.reward])
        nengo.Connection(reward_node, td_node[0])

        self.vals = nengo.Ensemble(nengo.LIF(1000), 2,
                              intercepts=Uniform(0, 1))
        nengo.Connection(input_node, self.vals)

        vals_relay = nengo.Node(size_in=num_actions)
        self.learn_conn = nengo.Connection(self.vals, vals_relay,
                                           function=lambda x: [0 for _ in range(num_actions)],
                                           learning_rule=nengo.PES(td_relay, learning_rate=l_rate),
                                           decoder_solver=decoder_solver,
                                           synapse=0.01)

        nengo.Connection(vals_relay, td_node[1:])

        def action_func(t, x):
            epsilon = 0.1
            mapping = [[-1], [0], [1]]
            if t % period <= 0.001:
                if random.random() < epsilon:
                    self.action = random.choice(mapping)
                else:
                    self.action = mapping[np.argmax(x)]
            return self.action
        action_node = nengo.Node(action_func, size_in=num_actions)
        nengo.Connection(vals_relay, action_node)
        nengo.Connection(action_node, output_node)

        freq = 0.1
        self.state_p = nengo.Probe(input_node, sample_every=freq)
        self.vals_p = nengo.Probe(vals_relay, sample_every=freq)
        self.reward_p = nengo.Probe(reward_node, sample_every=freq)
        self.err_p = nengo.Probe(td_node, sample_every=freq)


SEED = 0

nengo.log()

pong_game = PongGame(["computer", "computer"], seed=SEED)

d_file = "decoders.txt"
#d_solve = decoder_setter(d_file)
d_solve = nengo.decoders.lstsq_L2nz

with nengo.Network() as net:
    p0 = DecoderPongPlayer(0, pong_game)
    p1 = RLPongPlayer(1, pong_game, decoder_solver=d_solve,
                      l_rate=1e-5)


pong_game.start()
print "Pong started"

sim = nengo.Simulator(net, seed=SEED)
sim.run(1000)
print "done"

if isinstance(d_solve, decoder_setter):
    with open(d_file, "w") as f:
        f.write("\n".join([" ".join([str(x) for x in row]) for row in
                                    sim.signals[sim.model.sig_decoder[p1.learn_conn]].T]))


#print sim.data[p1.vals_p]
#print np.shape(sim.data[p1.vals_p]), np.shape(sim.trange(p1.vals_p.sample_every))
#print sim.signals['__time__'], sim.signals['__time__'] - sim.dt / 2
#print sim.trange(p1.vals_p.sample_every)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(sim.trange(p1.vals_p.sample_every), sim.data[p1.vals_p])

#plt.figure()
#plt.plot(sim.trange(p1.reward_p.sample_every), sim.data[p1.reward_p])
#
#plt.figure()
#plt.plot(sim.trange(p1.err_p.sample_every), sim.data[p1.err_p])

plt.figure()
enc = sim.model.params[p1.vals].encoders
dec = sim.signals[sim.model.sig_decoder[p1.learn_conn]].T
colours = [-1, 0, 1]
colour_array = [np.dot(colours, d) for d in dec]
plt.quiver(np.zeros(1000), np.zeros(1000), enc[:, 0], enc[:, 1],
           colour_array, scale=2.5)

plt.figure()
state = sim.data[p1.state_p]
pts = (state[:, 0] < -0.2) & (state[:, 0] > -0.5) & \
    (state[:, 1] > 0.2) & (state[:, 1] < 0.5)
plt.plot(sim.trange(p1.vals_p.sample_every)[pts],
         sim.data[p1.vals_p][pts])



plt.show()



