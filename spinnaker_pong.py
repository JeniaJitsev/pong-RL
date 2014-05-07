import math
import time
from pong_environment_play import PongGame

# ------------------------------------------
# Constants
# ------------------------------------------
DEBUG = True
MAX_Y = 480
UPDATE_TIMESTEP = 0.05          # s

NUM_PADDLE_CELLS = 4
PADDLE_CELL_SIGMA = 50.0        # In terms of max-y
PADDLE_CELL_MAX_RATE = 10.0     # Hz

NUM_BALL_CELLS = 4
BALL_CELL_SIGMA = 50.0          # In terms of max-y
BALL_CELL_MAX_RATE = 10.0       # Hz

# ------------------------------------------
# PlaceCell
# ------------------------------------------
# Very vague approximation of a place cell - spikes at a gaussian rate relative dependant on an input variable
class PlaceCell:
    def __init__(self, mu, sigma, max_rate, debug=False):
        self._mu = float(mu)
        self._two_sigma_squared = 2.0 * (float(sigma) ** 2)
        self._max_rate = float(max_rate)
        self._debug = debug
    
    def update(self, input_variable):
        # Calculate rate response using gaussian
        rate = self._max_rate * math.exp(-((float(input_variable) - self._mu) ** 2) / self._two_sigma_squared)
        if self._debug:
            print("\tCell mu %f, Rate %fHz" % (self._mu, rate))

# ------------------------------------------
# SpiNNakerPlayer
# ------------------------------------------
# Spinnaker AI using place cells to represent ball and paddle y position
class SpiNNakerPlayer:
    def __init__(self, player, env, max_y, num_paddle_cells, paddle_cell_sigma, paddle_cell_max_rate, num_ball_cells, ball_cell_sigma, ball_cell_max_rate, debug = False):
        self._debug = debug
        self._player = player
        self._env = env
        
        # Generate paddle cells
        if self._debug:
            print("Generating paddle place cells")
            
        self._paddle_cells = SpiNNakerPlayer._generate_evenly_spaced_place_cells(max_y, num_paddle_cells, paddle_cell_sigma, paddle_cell_max_rate, self._debug)
        
        # Generate ball cells
        if self._debug:
            print("Generating ball place cells")
            
        self._ball_cells = SpiNNakerPlayer._generate_evenly_spaced_place_cells(max_y, num_ball_cells, ball_cell_sigma, ball_cell_max_rate, self._debug)
    
    def update(self):
        # Get player state
        environmentState = self._env.getState(self._player)
        
        # Update paddle cells
        paddle_y = environmentState[1]
        if self._debug:
            print("Updating paddle cells: paddle_y %f" % paddle_y)
        
        for paddle_cell in self._paddle_cells:
            paddle_cell.update(paddle_y)
        
        # Update balls cells
        ball_y = self._env.state[0]
        if self._debug:
            print("Updating ball cells: ball_y %f" % ball_y)
        
        for ball_cell in self._ball_cells:
            ball_cell.update(ball_y)
            
    @staticmethod
    def _generate_evenly_spaced_place_cells(max_y, num_cells, cell_sigma, cell_max_rate, debug):
        assert num_cells > 1, "At least 2 cells are required"
        
        # Calculate even spacing between cells
        cell_spacing = float(max_y) / float(num_cells - 1)
        
        # Loop through cells
        cells = []
        for c in range(num_cells):
            # Calculate mu of this cell
            cell_mu = float(c) * cell_spacing
            
            # Add place cell to list
            if debug:
                print("\tPlace cell mu:%f" % cell_mu)
            
            cells.append(PlaceCell(cell_mu, cell_sigma, cell_max_rate, debug))
        
        return cells
   
# ------------------------------------------
# Entry point
# ------------------------------------------
pong_game = PongGame(["human", "computer"])
pong_game.start()

player_1  = SpiNNakerPlayer(0, pong_game, MAX_Y, NUM_PADDLE_CELLS, PADDLE_CELL_SIGMA, PADDLE_CELL_MAX_RATE, NUM_BALL_CELLS, BALL_CELL_SIGMA, BALL_CELL_MAX_RATE, DEBUG)
while True:
    time.sleep(UPDATE_TIMESTEP)
    player_1.update()


