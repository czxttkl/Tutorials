"""
Pocman environment first introduced in Monte-Carlo Planning in Large POMDPs by
Silver and Veness, 2010:
https://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps.pdf
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import itertools
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from typing import NamedTuple



MICRO = dict(
    _maze=np.array(
        [
            [1, 3, 3, 2, 3, 3],
            [3, 3, 0, 3, 0, 3],
            [3, 3, 3, 3, 3, 3],
            [1, 1, 0, 3, 0, 3],
            [1, 2, 3, 3, 3, 1],
        ],
        dtype=np.int8,
    ),
    _num_ghosts=2,
    _ghost_range=3,
    _ghost_home=(4, 4),
    _poc_home=(0, 0),
    _smell_range=1,
    _hear_range=2,
    _power_duration=15,
    _max_step=200,
)



STATE_DIM = 10


class Action:
    UP = 0
    RIGHT = 1  # east
    DOWN = 2
    LEFT = 3  # west


ACTIONS = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
ACTION_DICT = {Action.UP: "UP", Action.RIGHT: "RIGHT", Action.DOWN: "DOWN", Action.LEFT: "LEFT"}

class Element:
    WALL = 0
    CLEAR_WALK_WAY = 1
    POWER = 2
    FOOD_PELLET = 3


def random_action():
    return np.random.randint(0, 4)


def manhattan_distance(c1, c2):
    return np.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)


def opposite_direction(d):
    if d == Action.UP:
        return Action.DOWN
    if d == Action.RIGHT:
        return Action.LEFT
    if d == Action.DOWN:
        return Action.UP
    if d == Action.LEFT:
        return Action.RIGHT


class Position(NamedTuple):
    """ The index at the left up corner is (0, 0); The index at the right bottom corner is (height-1, width-1)"""
    x: int
    y: int


class InternalState(object):
    def __init__(self):
        self.agent_pos = None
        self.ghosts = []
        self.food_pos = []
        self.power_duration = 0


class Ghost(object):
    max_x: int
    max_y: int
    home: Position

    def __init__(self, env, pos, direction, ghost_range):
        self.env = env
        self.pos = pos
        self.direction = direction
        self.ghost_range = ghost_range
        self.move_type = "init"

    def move(self, agent_pos, agent_in_power):
        if manhattan_distance(agent_pos, self.pos) < self.ghost_range:
            if agent_in_power > 0:
                self.move_type = "defensive"
                self._move_defensive(agent_pos)
            else:
                self.move_type = "aggressive"
                self._move_aggressive(agent_pos)
        else:
            self.move_type = "random"
            self._move_random()


    def _move_random(self):
        movable_directions = set()
        for action in ACTIONS:
            next_pos = self.env.next_pos(self.pos, action)
            if self.pos != next_pos:
                movable_directions.add(action)

        if set([opposite_direction(self.direction), self.direction]) == movable_directions:
            next_pos = self.env.next_pos(self.pos, self.direction)
            self.update(next_pos, self.direction)
            return

        # no doubling back unless no other choice
        if opposite_direction(self.direction) in movable_directions and len(movable_directions) > 1:
            movable_directions.remove(opposite_direction(self.direction))
        d = np.random.choice(list(movable_directions))
        next_pos = self.env.next_pos(self.pos, d)
        self.update(next_pos, d)

    def _move_aggressive(self, agent_pos):
        best_dist = self.max_x + self.max_y
        best_pos = self.pos
        best_action = -1
        for a in ACTIONS:
            next_pos = self.env.next_pos(self.pos, a)
            # not movable in this action
            if next_pos == self.pos:
                continue
            dist = manhattan_distance(next_pos, agent_pos)
            if dist <= best_dist:
                best_pos = next_pos
                best_dist = dist
                best_action = a
        self.update(best_pos, best_action)

    def _move_defensive(self, agent_pos):
        best_dist = 0
        best_pos = self.pos
        best_action = -1
        for a in ACTIONS:
            next_pos = self.env.next_pos(self.pos, a)
            # not movable in this action
            if next_pos == self.pos:
                continue
            dist = manhattan_distance(next_pos, agent_pos)
            if dist >= best_dist:
                best_pos = next_pos
                best_dist = dist
                best_action = a
        self.update(best_pos, best_action)

    def update(self, pos, direction):
        self.pos = pos
        self.direction = direction

    def reset(self):
        self.pos = self.home
        # self.direction = random_action()


def select_maze(maze):
    maze = maze.lower()
    if maze == "micro":
        return MICRO
    else:
        raise NameError()


class PocEnv(Env):
    def __init__(self, maze):
        self.board = select_maze(maze)
        self._get_init_state()
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=1, shape=(8,))
        self._reward_range = 100
        self.step_cnt = 0

    def seed(self, seed=None):
        np.random.seed(seed)

    def _passable(self, pos):
        return self.maze[pos.x, pos.y] != Element.WALL

    def _inside(self, pos):
        if 0 <= pos.x < self.maze.shape[0] and 0 <= pos.y < self.maze.shape[1]:
            return True
        return False

    def step(self, action):
        assert self.action_space.contains(action)
        assert self.done is False
        self.step_cnt += 1

        reward = -1
        next_pos = self.next_pos(self.internal_state.agent_pos, action)
        self.internal_state.agent_pos = next_pos

        if self.internal_state.power_duration > 0:
            self.internal_state.power_duration -= 1
        agent_in_power = self.internal_state.power_duration > 0

        hit_ghost = set()
        for g, ghost in enumerate(self.internal_state.ghosts):
            # check if the ghost hits the agent before and after it moves
            if ghost.pos == self.internal_state.agent_pos:
                hit_ghost.add(g)
            else:
                ghost.move(self.internal_state.agent_pos, agent_in_power)
                if ghost.pos == self.internal_state.agent_pos:
                    hit_ghost.add(g)

        hit_ghost = list(hit_ghost)
        for g in hit_ghost:
            if self.internal_state.power_duration > 0:
                reward += 25
                self.internal_state.ghosts[g].reset()
            else:
                reward += -100
                self.done = True
                break

        if self.step_cnt > self.board["_max_step"]:
            self.done = True

        if self._agent_at_food():
            reward += 10
            self.maze[self.internal_state.agent_pos.x, self.internal_state.agent_pos.y] = Element.CLEAR_WALK_WAY
            if self._food_left() == 0:
                self.done = True

        if self._agent_at_power():
            self.internal_state.power_duration = self.board["_power_duration"]
            self.maze[self.internal_state.agent_pos.x, self.internal_state.agent_pos.y] = Element.CLEAR_WALK_WAY
            reward += 10

        ob = self._make_ob(action)

        return ob, reward, self.done, {"state": self.internal_state}

    def _agent_at_food(self):
        agent_pos = self.internal_state.agent_pos
        if self.maze[agent_pos.x, agent_pos.y] == Element.FOOD_PELLET:
            return True
        return False

    def _agent_at_power(self):
        agent_pos = self.internal_state.agent_pos
        if self.maze[agent_pos.x, agent_pos.y] == Element.POWER:
            return True
        return False

    def _make_ob(self, action):
        # 10 state features in total
        # 4 features indicating whether the agent can see a ghost
        # in that direction (UP, RIGHT, DOWN, LEFT)
        # 4 features indicating whether he can feel a wall in each of the cardinal
        # directions, which is set to 1 if he is adjacent to a wall.
        # 1 feature indicating whether he can hear a ghost, which is set to 1 if he is
        # within Manhattan distance 2 of a ghost.
        # 1 feature indicating whether he can smell food (adjacent or diagonally adjacent to any food)/
        ob = np.zeros(STATE_DIM)
        for i, action in enumerate(ACTIONS):
            if self._see_ghost(action):
                ob[i] = 1
            next_pos = self.next_pos(self.internal_state.agent_pos, action)
            # If an agent couldn't move from the current position, then there is a wall
            if next_pos == self.internal_state.agent_pos:
                ob[i + len(ACTIONS)] = 1
        if self._hear_ghost():
            ob[2*len(ACTIONS)] = 1
        if self._smell_food():
            ob[2*len(ACTIONS) + 1] = 1
        return ob

    def _see_ghost(self, action):
        agent_pos = self.internal_state.agent_pos
        for ghost in self.internal_state.ghosts:
            if agent_pos.x != ghost.pos.x and agent_pos.y != ghost.pos.y:
                continue
            if (action == Action.UP and ghost.pos.x < agent_pos.x) \
                or (action == Action.DOWN and ghost.pos.x > agent_pos.x) \
                or (action == Action.LEFT and ghost.pos.y < agent_pos.y) \
                or (action == Action.RIGHT and ghost.pos.y > agent_pos.y):
                if not self._wall_between(agent_pos, ghost.pos):
                    return True
        return False

    def _smell_food(self):
        smell_range = self.board["_smell_range"]
        agent_pos = self.internal_state.agent_pos

        for x in range(-smell_range, smell_range + 1):
            for y in range(-smell_range, smell_range + 1):
                smell_x = agent_pos.x + x
                smell_y = agent_pos.y + y
                if 0 <= smell_x < self.maze.shape[0] and 0 <= smell_y < self.maze.shape[1] and self.maze[smell_x, smell_y] == Element.FOOD_PELLET:
                    return True
        return False

    def _hear_ghost(self):
        for ghost in self.internal_state.ghosts:
            if manhattan_distance(ghost.pos, self.internal_state.agent_pos) <= self.board["_hear_range"]:
                return True
        return False

    def _wall_between(self, pos1, pos2):
        assert pos1 != pos2
        assert pos1.x == pos2.x or pos1.y == pos2.y
        if pos1.y == pos2.y:
            for i in range(min(pos1.x, pos2.x) + 1, max(pos1.x, pos2.x)):
                if self.maze[i, pos1.y] == Element.WALL:
                    return True
        elif pos1.x == pos2.x:
            for i in range(min(pos1.y, pos2.y), max(pos1.y, pos2.y)):
                if self.maze[pos1.x, i] == Element.WALL:
                    return True
        return False

    def _food_left(self):
        return np.sum(self.maze == Element.FOOD_PELLET)

    def reset(self):
        self.done = False
        self.step_cnt = 0
        self._get_init_state()
        return 0

    def _get_init_state(self):
        self.maze = self.board["_maze"].copy()
        self.internal_state = InternalState()
        self.internal_state.agent_pos = Position(*self.board["_poc_home"])
        Ghost.max_x = self.maze.shape[0]
        Ghost.max_y = self.maze.shape[1]
        ghost_home = Position(*self.board["_ghost_home"])
        Ghost.home = ghost_home

        for g in range(self.board["_num_ghosts"]):
            pos = Position(ghost_home.x, ghost_home.y)
            self.internal_state.ghosts.append(
                Ghost(self, pos, direction=random_action(), ghost_range=self.board["_ghost_range"]))

        return self.internal_state

    def next_pos(self, pos, action):
        x_offset, y_offset = 0, 0
        if action == Action.UP:
            x_offset = -1
            y_offset = 0
        elif action == Action.DOWN:
            x_offset = 1
            y_offset = 0
        elif action == Action.RIGHT:
            x_offset = 0
            y_offset = 1
        elif action == Action.LEFT:
            x_offset = 0
            y_offset = -1

        next_pos = Position(pos.x + x_offset, pos.y + y_offset)

        if self._inside(next_pos) and self._passable(next_pos):
            return next_pos
        else:
            return pos

    def print_internal_state(self):
        print_maze = self.maze.astype(str)
        print_maze[self.internal_state.agent_pos.x, self.internal_state.agent_pos.y] = 'A'
        ghost_str = ""
        for g, ghost in enumerate(self.internal_state.ghosts):
            print_maze[ghost.pos.x, ghost.pos.y] = 'G'
            ghost_str += "ghost {} at {}, direction={}, type={}\n".format(g, ghost.pos, ACTION_DICT[ghost.direction], ghost.move_type)
        np.set_printoptions(formatter={'str_kind': lambda x: x})
        print("maze: \n{}".format(print_maze))
        print("agent at {}, power duration {}".format(self.internal_state.agent_pos, self.internal_state.power_duration))
        print(ghost_str[:-1])

    def print_ob(self, ob):
        ob_str = ""
        for i, action in enumerate(ACTIONS):
            if ob[i] == 1:
                ob_str += " SEE GHOST {},".format(ACTION_DICT[action])
        for i, action in enumerate(ACTIONS):
            if ob[i + len(ACTIONS)] == 1:
                ob_str += " FEEL WALL {},".format(ACTION_DICT[action])
        if ob[-2]:
            ob_str += " HEAR GHOST,"
        if ob[-1]:
            ob_str += " SMELL FOOD,"
        return ob_str


if __name__ == "__main__":
    env = PocEnv("micro")
    env.seed(313)
    acc_rws = []

    for e in range(1000):
        env.reset()
        acc_rw = 0
        for i in itertools.count():
            print("step", i)
            env.print_internal_state()
            action = random_action()
            ob, rw, done, info = env.step(action)
            print("after action {}: reward {}, observation {} ({})".format(ACTION_DICT[action], rw, ob, env.print_ob(ob)))
            env.print_internal_state()
            acc_rw += rw
            if done:
                print("Epoch {} done in {} step, accumulated reward {}".format(e, i, acc_rw))
                print("\n\n\n")
                break
            print()
        acc_rws.append(acc_rw)

    print("average accumulated reward {}".format(np.mean(acc_rws)))
