import Queue
import threading
import pong

class PongGame(threading.Thread):
    def __init__(self, players, bins=480, seed=None):
        self.players = players
        self.seed = seed

        self.action_queue = [Queue.Queue(1) for _ in range(2)]
        self.action_lock = [threading.Lock() for _ in range(2)]

        self.reward_queue = [Queue.Queue(1) for _ in range(2)]
        self.reward_lock = [threading.Lock() for _ in range(2)]

        self.state_queue = Queue.Queue(4 * 1024)
        self.state_lock = threading.Lock()

        self.world_dim = {'ball_y':bins, 'paddle': bins}
        self.num_possible_moves = 3

        self.state = [1, 0, 0] # ball_y, paddle0, paddle1


        threading.Thread.__init__(self)

    def getWorldDim(self):
        return [self.world_dim['ball_y'], self.world_dim['paddle']]

    def getActionDim(self):
        return self.num_possible_moves

    def move(self, direction, player):
        if direction == 2:
            direction = -1 #convert to pong actions 

        self.action_lock[player].acquire()
        if not self.action_queue[player].empty():
            self.action_queue[player].get()
        self.action_queue[player].put(direction)
        self.action_lock[player].release()

        self.reward_lock[player].acquire()
        if not self.reward_queue[player].empty():
            r = self.reward_queue[player].get()
        else:
            r = 0
        self.reward_lock[player].release()

        return [self.getState(player), r]

    def get_stats(self):
        return pong.glob.hits + pong.glob.misses

    def update_state(self):
        self.state_lock.acquire()
        while not self.state_queue.empty():
            ball_x = self.state_queue.get()
            self.state[0] = min(self.world_dim['ball_y'] - 1, # -1 to avoid running off end of array
                                int(self.state_queue.get() / (480.0 / self.world_dim['ball_y'])))
            self.state[1] = min(self.world_dim['paddle'] - 1,
                                int(self.state_queue.get() / (480.0 / self.world_dim['paddle'])))
            self.state[2] = min(self.world_dim['paddle'] - 1,
                                int(self.state_queue.get() / (480.0 / self.world_dim['paddle'])))
        self.state_lock.release()

    def getState(self, player):
        self.update_state()
        self.state_lock.acquire()
        if player == 0:
            s = [self.state[0], self.state[1]]
        else:
            s = [self.state[0], self.state[2]]
        self.state_lock.release()
        return s

    def run(self):
        pong.main(self.players, self.action_lock, self.action_queue,
                  self.reward_lock, self.reward_queue, self.state_lock, self.state_queue,
                  seed=self.seed)




