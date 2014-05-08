#!/usr/bin/env python3

# Pong
# Written in 2013 by Julian Marchant <onpon4@riseup.net>
#
# To the extent possible under law, the author(s) have dedicated all
# copyright and related and neighboring rights to this software to the
# public domain worldwide. This software is distributed without any
# warranty.
#
# You should have received a copy of the CC0 Public Domain Dedication
# along with this software. If not, see
# <http://creativecommons.org/publicdomain/zero/1.0/>.

import sge
import Queue
import threading
import numpy.random as random

PADDLE_SPEED = 4
COMPUTER_PADDLE_SPEED = 2
PADDLE_VERTICAL_FORCE = 1 / 12
BALL_START_SPEED = 2
BALL_ACCELERATION = 0.0
BALL_MAX_SPEED = 15
SIM_STEP = 0.001



class glob:
    # This class is for global variables.  While not necessary, using a
    # container class like this is less potentially confusing than using
    # actual global variables.

    players = [None, None]
    ball = None
    hud_sprite = None
    bounce_sound = None
    bounce_wall_sound = None
    score_sound = None
    game_in_progress = True
    sim_time = 0.0

    hits = [0, 0]
    misses = [0, 0]

class Game(sge.Game):

    def event_key_press(self, key, char):
        if key == 'f8':
            sge.Sprite.from_screenshot().save('screenshot.jpg')
        elif key == 'escape':
            self.event_close()
        elif key in ('p', 'enter'):
            self.pause()

    def event_close(self):
        m = "Are you sure you want to quit?"
        if sge.show_message(m, ("No", "Yes")):
            self.end()

    def event_paused_key_press(self, key, char):
        if key == 'escape':
            # This allows the player to still exit while the game is
            # paused, rather than having to unpause first.
            self.event_close()
        else:
            self.unpause()

    def event_paused_close(self):
        # This allows the player to still exit while the game is paused,
        # rather than having to unpause first.
        self.event_close()

#    def event_step(self, t):
#        if glob.sim_time % 0.01 <= 0.001:
#            glob.hud_sprite.draw_clear()
#            glob.hud_sprite.draw_text("hud", "%.2f" % glob.sim_time, sge.game.width / 2,
#                                              0, color="white",
#                                              halign=sge.ALIGN_RIGHT,
#                                              valign=sge.ALIGN_TOP)


class ComputerPlayer(sge.StellarClass):
    lock = None
    queue = None

    state_lock = None
    state_queue = None
    reward_lock = None
    reward_queue = None

    def __init__(self, lock, queue, reward_lock, reward_queue, player_num):
        x = 32 if player_num == 0 else sge.game.width - 32
        y = sge.game.height / 2
        self.player_num = player_num
        self.hit_direction = 1 if player_num == 0 else -1
        self.lock = lock
        self.queue = queue
        self.reward_lock = reward_lock
        self.reward_queue = reward_queue
        super(ComputerPlayer, self).__init__(x, y, sprite="paddle_pc")

    def event_step(self, time_passed):
        move_direction = 0
        self.lock.acquire()
        while not self.queue.empty():
            move_direction = self.queue.get()
#            if self.player_num == 1: # don't want to double count
#                glob.sim_time += SIM_STEP
        self.lock.release()
        self.yvelocity = move_direction * COMPUTER_PADDLE_SPEED

        # Keep the paddle inside the window
        if self.bbox_top < 0:
            self.bbox_top = 0
        elif self.bbox_bottom > sge.game.height:
            self.bbox_bottom = sge.game.height

#        if self.y > sge.game.height:
#            self.y = 0
#        if self.y < 0:
#            self.y = sge.game.height

class Player(sge.StellarClass):

    def __init__(self, player_num):
        self.up_key = "up"
        self.down_key = "down"
        x = 32 if player_num == 0 else sge.game.width - 32
        self.player_num = player_num
        self.hit_direction = 1 if player_num == 0 else -1
        y = sge.game.height / 2
        super(Player, self).__init__(x, y, 0, sprite="paddle")

    def event_step(self, time_passed):
        # Movement
        key_motion = (sge.get_key_pressed(self.down_key) -
                      sge.get_key_pressed(self.up_key))

        self.yvelocity = key_motion * PADDLE_SPEED

        # Keep the paddle inside the window
        if self.y < 0:
            self.y = 0
        elif self.y > sge.game.height:
            self.y = sge.game.height




class Ball(sge.StellarClass):
    reward_lock = None
    reward_queue = None
    state_lock = None
    state_queue = None


    def __init__(self, reward_lock, reward_queue, state_lock, state_queue):
        x = sge.game.width / 2
        y = sge.game.height / 2
        self.reward_lock = reward_lock
        self.reward_queue = reward_queue
        self.state_lock = state_lock
        self.state_queue = state_queue

        super(Ball, self).__init__(x, y, 1, sprite="ball")

    def event_create(self):
        self.serve()

    def event_step(self, time_passed):
        # Scoring
        loser = None
        if self.bbox_right < 0:
            loser = 0
        elif self.bbox_left > sge.game.width:
            loser = 1

        if loser is not None:
            glob.misses[loser] += 1

            self.serve(1 if loser == 0 else -1)

            self.reward_lock[loser].acquire()
            if not self.reward_queue[loser].full():
#                self.reward_queue[loser].put(-abs(glob.ball.y - glob.players[loser].y) + 50)
                self.reward_queue[loser].put(-1)
            self.reward_lock[loser].release()


        # Bouncing off of the edges
        if self.bbox_bottom > sge.game.height:
            self.bbox_bottom = sge.game.height
            self.yvelocity = -abs(self.yvelocity) * 0.25
#            self.yvelocity = 0
        elif self.bbox_top < 0:
            self.bbox_top = 0
            self.yvelocity = abs(self.yvelocity) * 0.25
#            self.yvelocity = 0
#        if self.y > sge.game.height:
#            self.y = 0
#        if self.y < 0:
#            self.y = sge.game.height

        self.state_lock.acquire()
        if not self.state_queue.full():
            self.state_queue.put(glob.ball.x)
            self.state_queue.put(glob.ball.y)
            self.state_queue.put(glob.players[0].y)
            self.state_queue.put(glob.players[1].y)
        self.state_lock.release()


    def event_collision(self, other):
        if isinstance(other, (ComputerPlayer, Player)):
            if other.player_num == 0:
                self.bbox_left = other.bbox_right + 1
                self.xvelocity = min(abs(self.xvelocity) + BALL_ACCELERATION, BALL_MAX_SPEED)
                hitter = 0
            else:
                self.bbox_right = other.bbox_left - 1
                self.xvelocity = max(-abs(self.xvelocity) - BALL_ACCELERATION, -BALL_MAX_SPEED)
                hitter = 1
            self.yvelocity += (self.y - other.y) * (PADDLE_VERTICAL_FORCE + 0.01)

            glob.hits[hitter] += 1

            self.reward_lock[hitter].acquire()
            if not self.reward_queue[hitter].full():
                self.reward_queue[hitter].put(1)
            self.reward_lock[hitter].release()

    def serve(self, direction=1):
        self.x = sge.game.width / 2 + (200 if direction == -1 else -200)
        self.y = random.randint(0, sge.game.height)

        # Next round
        self.xvelocity = BALL_START_SPEED * direction
        self.yvelocity = 0



def main(players, action_lock, action_queue, reward_lock, reward_queue, state_lock, state_queue, seed=None):
    random.seed(seed)

    # Create Game object
    Game(640, 480, fps=120)

    # Load sprites
    paddle_sprite = sge.Sprite(ID="paddle", width=8, height=120, origin_x=4,
                               origin_y=60)
    paddle_sprite.draw_rectangle(0, 0, paddle_sprite.width,
                                 paddle_sprite.height, fill="white")

    paddle_sprite_pc = sge.Sprite(ID="paddle_pc", width=8, height=80, origin_x=4,
                                  origin_y=20)
    paddle_sprite_pc.draw_rectangle(0, 0, paddle_sprite.width,
                                 paddle_sprite.height, fill="white")


    ball_sprite = sge.Sprite(ID="ball", width=32, height=32, origin_x=16,
                             origin_y=16)
    ball_sprite.draw_rectangle(0, 0, ball_sprite.width, ball_sprite.height,
                               fill="white")

    glob.hud_sprite = sge.Sprite(width=320, height=160, origin_x=160,
                                 origin_y=0)
    hud = sge.StellarClass(sge.game.width / 2, 0, -10, sprite=glob.hud_sprite,
                           detects_collisions=False)

    # Load backgrounds
    layers = (sge.BackgroundLayer("ball", sge.game.width / 2, 0, -10000,
                                  xrepeat=False),)
    background = sge.Background (layers, "black")

    # Load fonts
    sge.Font('Liberation Mono', ID="hud", size=24)

    # Create objects
    for i in range(2):
        glob.players[i] = Player(i) if players[i] == "human" else \
                          ComputerPlayer(action_lock[i], action_queue[i],
                                         reward_lock[i], reward_queue[i], i)
    glob.ball = Ball(reward_lock, reward_queue, state_lock, state_queue)

    objects = glob.players + [glob.ball, hud]

    # Create rooms
    room1 = sge.Room(objects, background=background)

    sge.game.start()


if __name__ == '__main__':
    main()
