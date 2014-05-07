from pacman103.front.common.spike_source_poisson import SpikeSourcePoisson

import logging
logger = logging.getLogger(__name__)

class SpikeSourcePoissonDynamic(SpikeSourcePoisson):
    """
    This class represents a special Poisson Spike source object, which can represent
    a population of virtual neurons each with its own parameters. 
    """
    def __init__(self, atoms, contraints = None, label = "SpikeSourcePoisson",
            rate = 1, start = 0, duration = 10000, seed = None):
        """
        Creates a new SpikeSourcePoisson Object.
        """
        super( SpikeSourcePoissonDynamic, self ).__init__(
            atoms, contraints, label, rate, start, duration, seed
        )
    
    '''
    overloaded as dynamic poisson sources use multicast source to forward command words
    '''
    def requires_multi_cast_source(self):
        return True
