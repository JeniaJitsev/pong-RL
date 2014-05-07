from spike_source_poisson_dynamic import SpikeSourcePoissonDynamic
from pacman103.core.spinnman.sdp.sdp_connection import SDPConnection
from pacman103.core.spinnman.scp.scp_message import SCPMessage
import pacman103.front.pynn as sim
import threading
import time

# ------------------------------------------
# Constants
# ------------------------------------------
PADDLE_CELL_POP_SIZE = 50
BALL_CELL_POP_SIZE = 50
SIM_TIME = 100000
DEFAULT_RATE = 0.01
UPDATE_TIMESTEP = 1          # s

# ------------------------------------------
# PyNN entry point
# ------------------------------------------
# SpiNNaker setup
sim.setup(timestep=1.0,min_delay=1.0,max_delay=10.0)

class TxThread(threading.Thread):
    def __init__(self, board_address, update_timestep, dst_cpu, dst_x, dst_y):
        # Superclass
        super(TxThread, self).__init__()
        
        # Cache timsetep
        self._updateTimestep = update_timestep
        
        # Create SDP connection to hist
        self._connection = SDPConnection(board_address)
        
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
        self._message.dst_cpu = dst_cpu
        self._message.dst_x = dst_x
        self._message.dst_y = dst_y
        self._message.dst_port = (1 << 5) | 1
        
        # Set key message will be transmitted from
        # **TODO** get key from monitor vertex and combine with command
        self._message.arg1 = 0
        
        # Set that message has a payload
        # **THINK** is this actually correct
        self._message.cmd_rc = 1
        
    def run(self):
        while(True):
            # Sleep a bit
            time.sleep(self._updateTimestep)
       
            # **TEMP** set mean ISI to random number
            self._message.arg2 = TxThread._doubleToS1516(4.0)
            
            # Send message
            self._connection.send(self._message)
    
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
        

# Create two test cell populations
paddle_cell = sim.Population(PADDLE_CELL_POP_SIZE, SpikeSourcePoissonDynamic, {'rate': DEFAULT_RATE, 'start':0, 'duration':SIM_TIME})
ball_cell = sim.Population(PADDLE_CELL_POP_SIZE, SpikeSourcePoissonDynamic, {'rate': DEFAULT_RATE, 'start':0, 'duration':SIM_TIME})

# Create thread to transmit shit to board
thread = TxThread("192.168.240.253", UPDATE_TIMESTEP, 1, 0, 0)

# Start thread and simulation
thread.start()

#time.sleep(SIM_TIME)
sim.run(SIM_TIME)