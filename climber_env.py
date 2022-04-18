import gym
import climber_model
import routes
import pygame
import pygame_render
import numpy as np

_CHANGE_SUPPORT_ACTION = 666

class ClimberEnv(gym.Env):

    def __init__(self, climber: climber_model.Climber):
        self._climber = climber
        self._route = routes.generate_random_route(250, 250, 30)
        self._render_surface = pygame.display.set_mode((250, 250))
        self._final_hole = self._get_final_hole()
        
        self._init_action_space()
        self._init_observation_space()

        self._eposodes_count = 0
        self._eposodes_max = 50

        for st, _ in self._states.items():
            start_obs = st
            break

        self._start_left_hand_pos = np.array(start_obs[0])
        self._start_right_hand_pos = np.array(start_obs[1])
        self._start_left_leg_pos = np.array(start_obs[2])
        self._start_right_leg_pos = np.array(start_obs[3])
        self._start_support = start_obs[4]


    def step(self, action):
        self._eposodes_count += 1
        if (self._eposodes_count == self._eposodes_max - 1):
            return (self._curr_state, -120, True, "")

        decoded_action = self._actions[action]
        reward = -5
        done = False
        if decoded_action == _CHANGE_SUPPORT_ACTION:
            self._climber.change_support()
            reward = 2

        elif self._climber.is_transition_possible(limb=decoded_action["limb"], point=decoded_action["hole"]):
            self._climber.do_transition(limb=decoded_action["limb"], point=decoded_action["hole"])
            done = np.array_equal(self._climber.left_hand_pos, self._final_hole) \
                or np.array_equal(self._climber.right_hand_pos, self._final_hole)
            if done:
                reward = 140
            else:
                reward = 500 - min(np.linalg.norm(self._final_hole - self._climber.right_hand_pos), np.linalg.norm(self._final_hole - self._climber.left_hand_pos))
        
        self._curr_state = self._states[self._encode_obs(
            left_hand=self._climber.left_hand_pos,
            right_hand=self._climber.right_hand_pos,
            left_leg=self._climber.left_leg_pos,
            right_leg=self._climber.right_leg_pos,
            support=self._climber.support
        )]
        return (self._curr_state, reward, done, "")
             


    def reset(self):
        self._eposodes_count = 0
        self._climber.set_start_pos(
            left_hand=self._start_left_hand_pos,
            right_hand=self._start_right_hand_pos,
            left_leg=self._start_left_leg_pos,
            right_leg=self._start_right_leg_pos,
            support=self._start_support
        )
        self._curr_state = self._states[self._encode_obs(
            left_hand=self._start_left_hand_pos,
            right_hand=self._start_right_hand_pos,
            left_leg=self._start_left_leg_pos,
            right_leg=self._start_right_leg_pos,
            support=self._start_support
        )]
        return self._curr_state


    def render(self, mode='human'):
        self._render_surface.fill(pygame_render.BLACK)
        pygame_render.render_route(self._route, self._render_surface)
        pygame_render.render_hole(self._final_hole, self._render_surface, color=pygame_render.GREEN)
        pygame_render.render_climber(self._climber, self._render_surface)
        pygame.display.update()



    def close(self):
        pygame.quit()
        

    def _init_observation_space(self):
        self._states = {}
        counter = 0
        for left_hand in self._route:
            for right_hand in self._route:
                for left_leg in self._route:
                    for right_leg in self._route:
                        if self._climber.can_start(left_hand=left_hand, right_hand=right_hand, left_leg=left_leg, right_leg=right_leg):
                            self._states[self._encode_obs(left_hand, right_hand, left_leg, right_leg, climber_model.RIGHT_HAND_LEFT_LEG)] = counter
                            counter += 1
                            self._states[self._encode_obs(left_hand, right_hand, left_leg, right_leg, climber_model.LEFT_HAND_RIGH_LEG)] = counter
                            counter += 1

        self.observation_space = gym.spaces.Discrete(len(self._states))


    def _init_action_space(self):
        self._actions = {}
        counter = 0
        self._actions[counter] = _CHANGE_SUPPORT_ACTION
        counter += 1
        for limb in (climber_model.LEFT_HAND, climber_model.RIGHT_HAND, climber_model.LEFT_LEG, climber_model.RIGHT_LEG):
            for hole in self._route:
                self._actions[counter] = {
                    "limb": limb,
                    "hole": hole
                }
                counter += 1

        self.action_space = gym.spaces.Discrete(len(self._actions))


    def _get_final_hole(self):
        y_max = self._route[0][1]
        y_max_ind = 0
        for i in range(1, self._route.shape[0]):
            if y_max < self._route[i][1]:
                y_max = self._route[i][1]
                y_max_ind = i
        return self._route[y_max_ind]


    def _encode_obs(self, left_hand: np.ndarray, right_hand: np.ndarray, left_leg: np.ndarray, right_leg: np.ndarray, support: int):
        return (tuple(left_hand), tuple(right_hand), tuple(left_leg), tuple(right_leg), support)


    def _decode_obs(self, value: int):
        state = self._states[value]
        return {
            "left_hand": np.array(state[0]),
            "right_hand": np.array(state[1]), 
            "left_leg": np.array(state[2]), 
            "right_leg": np.array(state[3]), 
            "support": state[4],
            "done": state[5]
        }