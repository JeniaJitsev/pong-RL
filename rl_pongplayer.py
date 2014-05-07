import copy
import math

import numpy as np

import nengo
from nengo.utils.distributions import Uniform

class RLPongPlayer(nengo.Network):
    def __init__(self, player, env, decoder_solver=None, l_rate=1e-7,
                 discount=0.9, rng=None):
        num_actions = 3
        self.state = (0, 0)
        max_y = 480
        self.period = 0.1
        neuron = nengo.LIFRate
        self.place_dev = 0.15
        self.rng = rng
        self.stats = []
        self.threshold = 0.0
        self.reward = 0

        self.placecells = np.asarray(self.gen_placecells(min_spread=self.place_dev))
        print "placecells", len(self.placecells)


        def output_func(t, x):
            mapping = [-1, 0, 1]
            action = mapping[int(round(x[0]))]
            s, self.reward = env.move(action, player)
            self.state = [2.0 * x / max_y - 1 for x in s]

        def state_func(t):
            tmp = np.append(self.state, self.calc_activations(self.state, self.place_dev))
            return tmp

        input_node = nengo.Node(state_func)
        output_node = nengo.Node(output_func, size_in=1)

        td_node = nengo.Node(TDErrorCalc(self.period, num_actions, discount),
                             size_in=num_actions + 2, label="td_node")
        td_relay = nengo.Node(size_in=num_actions)
        nengo.Connection(td_node, td_relay)

        reward_node = nengo.Node(lambda t: [self.reward])
        nengo.Connection(reward_node, td_node[0])

        N = 1000
        self.vals = nengo.Ensemble(neuron(N), len(self.placecells),
                                  intercepts=Uniform(self.threshold, 1),
                                  encoders=self.gen_encoders(N),
                                  eval_points=[self.calc_activations(self.random_location(),
                                                                     self.place_dev)
                                               for _ in range(len(self.placecells) * 50)],
                                   radius=2)
        nengo.Connection(input_node[2:], self.vals)

        vals_relay = nengo.Node(size_in=num_actions)
        self.learn_conn = nengo.Connection(self.vals, vals_relay,
                                           function=lambda x: [0.1 for _ in range(num_actions)],
                                           learning_rule=nengo.PES(td_relay, learning_rate=l_rate),
                                           decoder_solver=decoder_solver,
                                           synapse=0.01)

        nengo.Connection(vals_relay, td_node[1:-1])

        def action_func(t, x):
            if t % self.period <= 0.001:
#                print "updating action", t

#                epsilon = 0.1
#                if rng.random() < epsilon:
#                    self.action = rng.randint(0, len(x) - 1)
#                else:
#                    self.action = np.argmax(x)

                es = np.exp(x) / sum(np.exp(x))
                pick = rng.random()
                for i, e in enumerate(es):
                    pick -= e
                    if pick <= 0:
                        self.action = i
                        break

                self.stats.append(env.get_stats())
            return self.action
        action_node = nengo.Node(action_func, size_in=num_actions)
        nengo.Connection(vals_relay, action_node)
        nengo.Connection(action_node, output_node)
        nengo.Connection(action_node, td_node[4])

        freq = self.period / 2
        self.state_p = nengo.Probe(input_node, sample_every=freq)
        self.vals_p = nengo.Probe(vals_relay, sample_every=freq)
#        self.reward_p = nengo.Probe(reward_node, synapse=2 * freq, sample_every=freq)
        self.err_p = nengo.Probe(td_node, sample_every=freq)
        self.val_spikes = nengo.Probe(self.vals.neurons, "output", sample_every=0.1)

    def calc_activations(self, loc, place_dev):
        dists = np.sqrt(np.sum((self.placecells - np.array(loc)) ** 2, axis=1))
        return np.exp(-dists ** 2 / (2 * place_dev ** 2))

    def gen_placecells(self, min_spread=0.2):
        """Generate the place cell locations that will give rise to the state representation.

        :param min_spread: the minimum distance between place cells
        """

        N = None
        num_tries = 1000 # a limit on the number of attempts to place a new placecell

        # assign random x,y locations to each neuron
        locations = [self.random_location()]
        while True:
            # generate a random new point
            new_loc = self.random_location()

            # check that the point isn't too close to previous points
            count = 0
            while min([self.calc_dist(new_loc, l) for l in locations]) < min_spread and count < num_tries:
                new_loc = self.random_location()
                count += 1

            # add the new point
            locations += [new_loc]

            if (N == None and count >= num_tries) or len(locations) == N:
                # stop when required number of place cells built (if N specified),
                # or when world has been decently filled
                break

        return locations

    def random_location(self, radius=1):
        return (self.rng.random() * 2 * radius - radius,
                self.rng.random() * 2 * radius - radius)

    def gen_encoders(self, N):
        """Generate encoders for state population in RL agent."""

        locs = self.placecells

        encoders = [None for _ in range(N)]
        for i in range(N):
            # pick a random point for the neuron
            pt = self.random_location() # could make this avoid walls if we want

            # set the encoder to be the inverse of the distance from each placecell to that point
            encoders[i] = [1.0 / self.calc_dist(pt, l) for l in locs]

            # cut off any values below a certain threshold
            encoders[i] = [x if x > 0.5 * max(encoders[i]) else 0 for x in encoders[i]]

            # normalize the encoder
            encoders[i] = [x / math.sqrt(sum([y ** 2 for y in encoders[i]])) for x in encoders[i]]

        return encoders

    def calc_dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class TDErrorCalc:
    def __init__(self, period, num_actions, discount):
        self.period = period
        self.saved_vals = [0 for _ in range(num_actions)]
        self.saved_action = 0
        self.reward = 0
        self.discount = discount

    def __call__(self, t, x):
        self.reward += x[0]

        if t % self.period <= 0.001:
#            print "calculating error", t
            vals = x[1:-1]
            self.saved_action = int(round(x[-1]))
                # note: this node gets updated before the action node, so this action
                # represents the action selected in the previous time period

            err = self.reward + self.discount * max(vals) - self.saved_vals[self.saved_action]
            self.err_sig = [err if i == self.saved_action else
                            (0.001 if vals[i] < 0 else 0)
                            for i in range(len(vals))]
#            print "err_sig", self.err_sig

            print "%.1f % .3f % .3f % .3f % .3f %d" % (t, err, self.reward,
                                           self.discount * max(vals),
                                           self.saved_vals[self.saved_action],
                                           self.saved_action)

            self.reward *= 0.6 # note: doing a slow decay rather than setting to 0, to put
                                # a bit more power in the reward signal
            self.saved_vals = copy.copy(vals)

        return self.err_sig

