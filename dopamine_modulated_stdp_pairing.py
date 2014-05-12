import numpy, pylab, random, sys
import NeuroTools.signals as nt

from dopamine_modulated_spike_pair import DopamineModulatedSpikePairRule
from if_curr_exp_dopamine import IF_CurrentExponentialDopaminePopulation
import pacman103.front.pynn as sim

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0)

# +-------------------------------------------------------------------+
# | General Parameters                                                |
# +-------------------------------------------------------------------+

# Population parameters
lif_cell_params = {
    'cm'                : 0.25, # nF
    'i_offset'          : 0.0,
    'tau_m'             : 10.0,
    'tau_refrac'        : 2.0,
    'tau_syn_E'         : 2.5,
    'tau_syn_I'         : 2.5,
    'v_reset'           : -70.0,
    'v_rest'            : -65.0,
    'v_thresh'          : -55.4
}

dopamine_lif_cell_params = {
    'cm'                : 0.25, # nF
    'i_offset'          : 0.0,
    'tau_m'             : 10.0,
    'tau_refrac'        : 2.0,
    'tau_syn_E'         : 2.5,
    'tau_syn_I'         : 2.5,
    'tau_syn_dopamine'  : 200.0,
    'v_reset'           : -70.0,
    'v_rest'            : -65.0,
    'v_thresh'          : -55.4
}

# Other simulation parameters
e_rate = 200
in_rate = 350


delta_t = 10
time_between_pairs = 150
num_pre_pairs = 10
num_pairs = 100
num_post_pairs = 10
pop_size = 1


pairing_start_time = (num_pre_pairs * time_between_pairs) + delta_t
pairing_end_time = pairing_start_time + (num_pairs * time_between_pairs)
sim_time = pairing_end_time + (num_post_pairs * time_between_pairs)


# +-------------------------------------------------------------------+
# | Creation of neuron populations                                    |
# +-------------------------------------------------------------------+
# Neuron populations

pre_pop = sim.Population(pop_size, sim.IF_curr_exp, lif_cell_params)
post_pop = sim.Population(pop_size, IF_CurrentExponentialDopaminePopulation, dopamine_lif_cell_params)
post_pop_dope = sim.Population(pop_size, IF_CurrentExponentialDopaminePopulation, dopamine_lif_cell_params)


# Stimulating populations
pre_stim = sim.Population(pop_size, sim.SpikeSourceArray, {'spike_times': [[i for i in range(0, sim_time, time_between_pairs)],]})
post_stim = sim.Population(pop_size, sim.SpikeSourceArray, {'spike_times': [[i for i in range(pairing_start_time, pairing_end_time, time_between_pairs)],]})

# Dopamine source population
dopamine_source = sim.Population(pop_size, sim.SpikeSourcePoisson, {'rate': 20.0 , 'start' : 0, 'duration' : sim_time})

# +-------------------------------------------------------------------+
# | Creation of connections                                           |
# +-------------------------------------------------------------------+
# Connection type between noise poisson generator and excitatory populations
ee_connector = sim.OneToOneConnector(weights=2)

# Connect stimulus to pre and post populations
sim.Projection(pre_stim, pre_pop, ee_connector, target='excitatory')
sim.Projection(post_stim, post_pop, ee_connector, target='excitatory')
sim.Projection(post_stim, post_pop_dope, ee_connector, target='excitatory')

# Connect dopamine source to population's dopamine synapses
sim.Projection(dopamine_source, post_pop_dope, ee_connector, target='dopamine')

# Plastic Connections between pre_pop and post_pop
stdp_model = sim.STDPMechanism(
  timing_dependence = DopamineModulatedSpikePairRule(tau_plus = 20.0, tau_minus = 50.0),
  weight_dependence = sim.AdditiveWeightDependence(w_min = 0, w_max = 2, A_plus=0.02, A_minus = 0.02)
)

sim.Projection(pre_pop, post_pop_dope, sim.OneToOneConnector(), target='excitatory',
  synapse_dynamics = sim.SynapseDynamics(slow= stdp_model)
)

sim.Projection(pre_pop, post_pop, sim.OneToOneConnector(), target='excitatory',
  synapse_dynamics = sim.SynapseDynamics(slow= stdp_model)
)

# Record spikes
pre_stim.record()
post_stim.record()
pre_pop.record()
post_pop.record()
post_pop_dope.record()
dopamine_source.record()

# Run simulation
sim.run(sim_time)

def plot_spikes(spikes, axis, title):
  axis.set_xlim([0, sim_time])
  axis.set_xlabel('Time/ms')
  axis.set_ylabel('spikes')
  axis.set_title(title)
  
  if spikes != None:
      axis.plot([i[1] for i in spikes], [i[0] for i in spikes], ".") 
  else:
      print "No spikes received"

pre_spikes = pre_pop.getSpikes(compatible_output=True)
post_spikes = post_pop.getSpikes(compatible_output=True)
pre_stim_spikes = pre_stim.getSpikes(compatible_output=True)
post_stim_spikes = post_stim.getSpikes(compatible_output=True)
post_dope_spikes = post_pop_dope.getSpikes(compatible_output=True)
dopamine_source_spikes = dopamine_source.getSpikes(compatible_output=True)

figure, axisArray = pylab.subplots(3, 2)

plot_spikes(pre_spikes, axisArray[0, 1], "Pre-synaptic neurons")
plot_spikes(post_spikes, axisArray[1, 1], "Post-synaptic neurons low DA")
plot_spikes(pre_stim_spikes, axisArray[0, 0], "Pre-synaptic stimulus")
plot_spikes(post_stim_spikes, axisArray[1, 0], "Post-synaptic stimulus")
plot_spikes(dopamine_source_spikes, axisArray[2, 0], "Dopamine stimulus")
plot_spikes(post_dope_spikes, axisArray[2, 1], "Post-synaptic neuron high DA")

pylab.show()


# End simulation on SpiNNaker
sim.end()
