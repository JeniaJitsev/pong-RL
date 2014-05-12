import itertools
import math
import time
import threading
import visualiser.visualiser_modes
from pong_environment_play import PongGame
from spike_source_poisson_dynamic import SpikeSourcePoissonDynamic
from pacman103.core.spinnman.sdp.sdp_connection import SDPConnection
from pacman103.core.spinnman.scp.scp_message import SCPMessage
import pacman103.front.pynn as sim

# ------------------------------------------
# Constants
# ------------------------------------------
DEBUG = False
BOARD_ADDRESS = "192.168.240.253"
MAX_Y = 480
SIM_TIME = 100000
UPDATE_TIMESTEP = 0.1          # s

CELL_MIN_RATE = 0.01

NUM_PADDLE_CELLS = 2
PADDLE_CELL_SIGMA = 100.0        # In terms of max-y
PADDLE_CELL_MAX_RATE = 100.0     # Hz
PADDLE_CELL_POP_SIZE = 10

NUM_BALL_CELLS = 2
BALL_CELL_SIGMA = 100.0          # In terms of max-y
BALL_CELL_MAX_RATE = 100.0       # Hz
BALL_CELL_POP_SIZE = 10

INPUT_LAYER_POP_SIZE = 10

# ------------------------------------------
# DynamicPoissonSource
# ------------------------------------------
class DynamicPoissonSource:
    def __init__(self, population_size, population_label, min_rate, sdp_connection):
        self._min_rate = float(min_rate)
        self._sdp_connection = sdp_connection
        self._population_label = population_label
        
        # Create SDP message 
        self._message = SCPMessage()
        
        self._message.flags = 0x7
        self._message.tag = 255
        
        # Set source as ethernet core
        self._message.src_cpu = 0
        self._message.src_x = 0
        self._message.src_y = 0
        self._message.src_port = 255
        
        # Set destination
        self._message.dst_port = (1 << 5) | 1
        
        # Create population
        # **TODO** check if there is proper handling for non-finite durations in ssp
        self._population = sim.Population(population_size, sim.SpikeSourcePoisson, {'rate': self._min_rate, 'start' : 0, 'duration' : SIM_TIME}, label = self._population_label)
        
        # Record spikes
        # **TODO** make optional
        self._population.record(visuliser_mode = visualiser.visualiser_modes.RASTER)
    
    def setRate(self, rate):
        # Clamp rate
        rate = max(rate, self._min_rate)
        
        # Set argument 1 to command and argument 2 to the desired rate
        self._message.arg1 = 0
        self._message.arg2 = DynamicPoissonSource._doubleToS1516(float(rate))
        
        # Loop through target spike source's sub vertices
        for subvertex in self._population.vertex.subvertices:
            # Get sub-vertex's placement and set as message destination
            (x, y, p) = subvertex.placement.processor.get_coordinates()
            self._message.dst_cpu = p
            self._message.dst_x = x
            self._message.dst_y = y
            
            if DEBUG:
                print("Sending command to %u %u %u" % (x, y, p))
            
            # Send message
            self._sdp_connection.send(self._message)
        
    @staticmethod
    def _doubleToS1516(double):
        """
        Reformat a double into a 32-bit integer representing s1615 format
        (i.e. signed 16.15).
        Raise an exception if the value cannot be represented in this way.
        """
        if (double < -65536) or (double > (65536.0-1/32768.0)):
            raise Exception("ERROR: DSG - double cannot be recast as a s1615. Exiting.")

        # Shift up by 15 bits:
        scaledMyDouble = double * 32768.0

        # Round to an integer:
        # **THINK** should we actually round here?
        myS1615 = int(scaledMyDouble)
        return myS1615
    
    @property
    def population(self):
        return self._population
    
    @property
    def population_label(self):
        return self._population_label
        
        
# ------------------------------------------
# PlaceCell
# ------------------------------------------
# Very vague approximation of a place cell - spikes at a gaussian rate relative dependant on an input variable
class PlaceCell:
    def __init__(self, mu, sigma, min_rate, max_rate, population_size, population_label, sdp_connection):
        self._mu = float(mu)
        self._two_sigma_squared = 2.0 * (float(sigma) ** 2)
        self._max_rate = float(max_rate)
        
        # Create population
        self._poisson_source = DynamicPoissonSource(population_size, population_label, min_rate, sdp_connection)
        
    def update(self, input_variable):
        # Calculate rate response using gaussian
        rate = self._max_rate * math.exp(-((float(input_variable) - self._mu) ** 2) / self._two_sigma_squared)
        
        if DEBUG:
            print("\tCell mu %f, Rate %fHz" % (self._mu, rate))
        
        # Update poisson source
        self._poisson_source.setRate(rate)
    
    @property
    def population_label(self):
        return self._poisson_source.population_label
    
    @property
    def population(self):
        return self._poisson_source.population

# ------------------------------------------
# SpiNNakerPlayer
# ------------------------------------------
# Spinnaker AI using place cells to represent ball and paddle y position
class SpiNNakerPlayer(threading.Thread):
    def __init__(self, board_address, player, env, max_y, update_timestep, 
                 num_paddle_cells, paddle_cell_sigma, paddle_cell_min_rate, paddle_cell_max_rate, paddle_cell_population_size, 
                 num_ball_cells, ball_cell_sigma, ball_cell_min_rate, ball_cell_max_rate, ball_cell_population_size):
        # Superclass
        super(SpiNNakerPlayer, self).__init__()
        
        # Cache player and environment
        self._player = player
        self._env = env
        self._update_timestep = update_timestep
        self._input_layer_populations = []
        
        # Create SDP connection to allow place cells to communicate with board
        self._connection = SDPConnection(board_address)
        
        # Generate paddle cells
        if DEBUG:
            print("Generating paddle place cells")
            
        self._paddle_cells = SpiNNakerPlayer._generate_evenly_spaced_place_cells(max_y, num_paddle_cells, paddle_cell_sigma, paddle_cell_min_rate, paddle_cell_max_rate, paddle_cell_population_size, "paddle_cell", self._connection)
        
        # Generate ball cells
        if DEBUG:
            print("Generating ball place cells")
            
        self._ball_cells = SpiNNakerPlayer._generate_evenly_spaced_place_cells(max_y, num_ball_cells, ball_cell_sigma, ball_cell_min_rate, ball_cell_max_rate, ball_cell_population_size, "ball_cell", self._connection)
        '''
        # Loop through all cartesian products of paddle and ball cells to create a hardwired layer with higher dimension
        for input_layer_pair in itertools.product(self._paddle_cells, self._ball_cells):
            paddle_cell = input_layer_pair[0]
            ball_cell = input_layer_pair[1]
            
            # Create a population
            self._input_layer_populations.append(sim.Population(input_layer_population_size, input_layer_model, input_layer_parameters, label = "input_%s_%s" % (paddle_cell.population_label, ball_cell.population_label)))
            
            # Create a projection from each place cell to new population
        '''
            
        
    def run(self):
        while(True):
            # Get player state
            environmentState = self._env.getState(self._player)
            
            # Update paddle cells
            paddle_y = environmentState[1]
            if DEBUG:
                print("Updating paddle cells: paddle_y %f" % paddle_y)
            
            for paddle_cell in self._paddle_cells:
                paddle_cell.update(paddle_y)
            
            # Update balls cells
            ball_y = self._env.state[0]
            if DEBUG:
                print("Updating ball cells: ball_y %f" % ball_y)
             
            for ball_cell in self._ball_cells:
                ball_cell.update(ball_y)
            
            # Sleep a bit
            time.sleep(self._update_timestep)
            
    @staticmethod
    def _generate_evenly_spaced_place_cells(max_y, num_cells, cell_sigma, cell_min_rate, cell_max_rate, population_size, population_label_stem, sdp_connection):
        assert num_cells > 1, "At least 2 cells are required"
        
        # Calculate even spacing between cells
        cell_spacing = float(max_y) / float(num_cells - 1)
        
        # Loop through cells
        cells = []
        for c in range(num_cells):
            # Calculate mu of this cell
            cell_mu = float(c) * cell_spacing
            
            # Add place cell to list
            if DEBUG:
                print("\tPlace cell mu:%f" % cell_mu)
            
            cells.append(PlaceCell(cell_mu, cell_sigma, cell_min_rate, cell_max_rate, population_size, population_label_stem + ("_%u" % c), sdp_connection))
        
        return cells
    
# ------------------------------------------
# PyNN entry point
# ------------------------------------------
# SpiNNaker setup
sim.setup(timestep=1.0,min_delay=1.0,max_delay=10.0)

# Create human vs ai pong game and start it
pong_game = PongGame(["human", "computer"])
pong_game.start()

# Create spinnaker pong player
player_thread = SpiNNakerPlayer(BOARD_ADDRESS, 0, pong_game, MAX_Y, UPDATE_TIMESTEP, 
                NUM_PADDLE_CELLS, PADDLE_CELL_SIGMA, CELL_MIN_RATE, PADDLE_CELL_MAX_RATE, PADDLE_CELL_POP_SIZE, 
                NUM_BALL_CELLS, BALL_CELL_SIGMA, CELL_MIN_RATE, BALL_CELL_MAX_RATE, BALL_CELL_POP_SIZE)


# Start player thread and simulation
player_thread.start()

sim.run(SIM_TIME)

