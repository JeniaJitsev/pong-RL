from pacman103.front.pynn.synapse_dynamics.spike_pair_rule import SpikePairRule

class DopamineModulatedSpikePairRule(SpikePairRule):
    def __init__(self, tau_plus = 20.0, tau_minus = 20.0, base_dopamine_level = 0.5, dopamine_multiplier = 1.0):
        self.base_dopamine_level = base_dopamine_level
        self.dopamine_multiplier = dopamine_multiplier
        
        # Superclass
        # **TODO** add nearest neighbour version of dopamine moduled spike
        super(DopamineModulatedSpikePairRule, self).__init__(tau_plus, tau_minus, False)
        
    def __eq__(self, other):
        if super(DopamineModulatedSpikePairRule, self).__eq__(other):
            return ((self.base_dopamine_level == other.base_dopamine_level) and (self.dopamine_multiplier == other.dopamine_multiplier))
        else:
            return False
        
    def write_plastic_params(self, spec, machineTimeStep, subvertex):
        # Write parameters
        spec.write(data = spec.doubleToS1615(self.base_dopamine_level), sizeof = 's1615')
        spec.write(data = spec.doubleToS1615(self.dopamine_multiplier), sizeof = 's1615')
        
        # Superclass
        super(DopamineModulatedSpikePairRule, self).write_plastic_params(spec, machineTimeStep, subvertex)
        
    def get_params_size_bytes(self):
        return super(DopamineModulatedSpikePairRule, self).get_params_size_bytes() + (4 * 2)

    def get_vertex_executable_suffix(self):
        return "pair"