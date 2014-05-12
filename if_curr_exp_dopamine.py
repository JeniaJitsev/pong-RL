import math

from pacman103.lib import data_spec_constants
from pacman103.front.common.exp_population_vertex import ExponentialPopulationVertex
from pacman103.front.common.neuron_parameter import NeuronParameter

class IF_CurrentExponentialDopaminePopulation( ExponentialPopulationVertex ):
    core_app_identifier = data_spec_constants.IF_CURR_EXP_CORE_APPLICATION_ID

    def __init__( self, n_neurons, constraints = None, label = None,
                  tau_m = 20, cm = 1.0, 
                  v_rest = -65.0, v_reset = -65.0, v_thresh = -50.0,
                  tau_syn_E = 5.0, tau_syn_dopamine = 5.0, tau_syn_I = 5.0, tau_refrac = 0.1,
                  i_offset = 0, v_init = None):
        # Instantiate the parent class
        super( IF_CurrentExponentialDopaminePopulation, self ).__init__(
            n_neurons = n_neurons,
            n_params = 10,
            binary = "IF_curr_exp_dopamine.aplx",
            synapse_types = [("excitatory", float(tau_syn_E)), ("inhibitory", float(tau_syn_I)), ("dopamine", tau_syn_dopamine)],
            constraints = constraints,
            label = label
        )

        # Save the parameters
        self.tau_m = float(tau_m)
        self.cm = float(cm)
        self.v_rest = float(v_rest)
        self.v_reset = float(v_reset)
        self.v_thresh = float(v_thresh)
        self.tau_refrac = float(tau_refrac)
        self.i_offset = float(i_offset)
        self.v_init = float(v_rest)
        if v_init is not None:
            self.v_init = float(v_init)
        self.ringbuffer_saturation_scaling = 32 # Max accumulated ringbuffer value
                                                # in 'weight units' used to scale
                                                # weight value to a fraction

    def initialize_v( self, value):
        self.v_init = value
        
    @property
    def r_membrane( self ):
        return self.tau_m / self.cm

    def exp_tc( self, machineTimeStep ):
        return math.exp(
            - float(machineTimeStep) / (
                1000.0 * self.tau_m
            )
        )

    @property
    def one_over_tauRC( self ):
        """TODO: Scaling for when machine timestep =/= 1ms."""
        return 1.0 / self.tau_m

    @property
    def refract_timer( self ):
        return 0

    @property
    def t_refract( self ):
        return self.tau_refrac

    @property
    def model_name( self ):
        return "IF_curr_exp_dopamine"
    
    def getCPU(self, lo_atom, hi_atom):
        """
        Gets the CPU requirements for a range of atoms
        """
        return 782 * ((hi_atom - lo_atom) + 1)
    
    def get_maximum_atoms_per_core(self):
        '''
        returns the maxiumum number of atoms that a core can support
        for this model
        '''
        return 256

    def get_parameters(self, machineTimeStep):
        """
        Generate Neuron Parameter data (region 2):
        """

        # Get the parameters
        return [
            NeuronParameter( self.v_thresh,                's1615',   1.0 ),
            NeuronParameter( self.v_reset,                 's1615',   1.0 ),
            NeuronParameter( self.v_rest,                  's1615',   1.0 ),
            NeuronParameter( self.r_membrane,              's1615',   1.0 ),
            NeuronParameter( self.v_init,                  's1615',   1.0 ),
            NeuronParameter( self.i_offset,                's1615',   1.0 ),
            NeuronParameter( self.exp_tc(machineTimeStep), 's1615',   1.0 ),
            NeuronParameter( self.one_over_tauRC,          's1615',   1.0 ),
            NeuronParameter( self.refract_timer,           'uint32',  1.0 ),
            NeuronParameter( self.t_refract,               'uint32', 10.0 ),
        ]
