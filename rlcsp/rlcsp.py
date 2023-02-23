#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:41:17 2020
CSP variation of Reinforce algorithm designed to work with basin-hopping CSP codes
on the step where the next action to change the trial structure is being selected
@author: Elena Zamaraeva
"""
import numpy as np
import math
import statistics
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import os
from .state import State
from matplotlib.animation import FuncAnimation, PillowWriter
import time
import random
from mysql.connector import connect, Error
import string
import scipy.stats
import heapq
# from AppKit import NSSound

class Reinforce:

    unscaled = 'unscaled'
    change_in_features = 'features'
    uniqueness = 'uniqueness'

    f_energy = 'energy'
    f_sl = 'sl'
    f_old_energy = 'old_energy'

    def __init__(self, actions=["T1", "T2", "T3", "T4", "T5", "T6"],
                 alpha=0,
                 params_db={'host':'localhost',
                            'database':'',
                            'user':'',
                            'password':''},
                 reinforce_table='reinforce',
                 theta_table='theta',
                 reward_type='',
                 sl_range=[5,10],
                 features_set=['energy'],
                 episode_length=1,
                 fixed_episodes=True,
                 max_energy=0,
                 reinforce_id=0,
                 db_max_attempts=5,
                 debug=False,
                 reg_params={'e_threshold': 0,
                             'beta': 0,
                             'free_term': 0,
                             'h_type': 'non-linear',
                             'theta':'',
                             'zero_reward_penalty': 0,
                             'non_converge_penalty': 0,
                             'non_unique_penalty': 0,
                             'scale_reward': False,
                             'epsilon': 0,
                             'non_unique_reward': True,
                             'step_reward_limit': 100,
                             'reward_limit_last': False, # if True we take the _last_ step_reward_limit rewards
                                                        # for the reward scaling
                             'smart_penalty': False}): # if True we calculate the penalties as the mean of smalles rewards

        self.reward_type = Reinforce.change_in_features if reward_type == '' else reward_type
        if 'zero_reward_penalty' not in reg_params:
            self.zero_reward_penalty = 0
        else:
            self.zero_reward_penalty = reg_params['zero_reward_penalty']
        if 'non_converge_penalty' not in reg_params:
            self.non_converge_penalty = 0
        else:
            self.non_converge_penalty = reg_params['non_converge_penalty']
        if 'non_unique_penalty' not in reg_params:
            self.non_unique_penalty = 0
        else:
            self.non_unique_penalty = reg_params['non_unique_penalty']

        self.max_energy = max_energy

        self.beta = 1

        # regularization parameters: e_threshold is the entropy threshold after which we apply regularization
        # beta is the coefficient of confidence penalty
        self.reg_params = reg_params
        if 'e_threshold' not in self.reg_params:
            self.reg_params['e_threshold'] = 0
        if 'epsilon' not in self.reg_params:
            self.reg_params['epsilon'] = 0
        if 'beta' not in self.reg_params:
            self.reg_params['beta'] = 0
        if 'scale_reward' not in self.reg_params:
            self.reg_params['scale_reward'] = False
        if 'non_unique_reward' not in self.reg_params:
            self.reg_params['non_unique_reward'] = True
        if 'reward_limit_last' not in self.reg_params:
            self.reg_params['reward_limit_last'] = False
        if 'smart_penalty' not in self.reg_params:
            self.reg_params['smart_penalty'] = False
        self.free_term = self.reg_params['free_term']
        self.h_type = self.reg_params['h_type']

        self.episode_length = episode_length
        self.fixed_episodes = fixed_episodes
        self.episode = []

        self.id = reinforce_id
        self.db_max_attempts = db_max_attempts

        self.actions = actions
        self.theta = [0 for k in range(len(features_set)*len(self.actions)*2)]
        # a_prob = {'T1': 1.8, 'T2': 1.47, 'T3': 1.47, 'T4': 1.07, 'T5': 1.07, 'T6': 0.4}
        # a_prob = {'T1': 15.2, 'T2': 12, 'T3': 12, 'T4': 8, 'T5': 1, 'T6': 1}
        # a_prob = {'T1': 1, 'T2': 1, 'T3': 1, 'T4': 1, 'T5': 1, 'T6': 1}
        # a_prob = {'T1': 3.04, 'T2': 2.4, 'T3': 2.4, 'T4': 1.6, 'T5': 0.2, 'T6': 0.2}
        # for i in range(len(self.actions)):
        #     self.theta[i*2 + 1] = a_prob[self.actions[i]]

        self.thetas = []
        self.f_array = {f: [] for f in features_set}
        self.f_mean = {f: None for f in features_set}
        self.f_std = {f: None for f in features_set}
        self.f_min = {f: None for f in features_set}
        self.f_max = {f: None for f in features_set}

        self.mean_reward = None
        self.std_reward = None

        self.debug = debug

        # self.rewards_change = []
        # self.rewards_standard_change = []

        if Reinforce.f_sl in features_set:
            self.f_min[Reinforce.f_sl] = min(sl_range)
            self.f_max[Reinforce.f_sl] = max(sl_range)

        # self.energies = []
        # self.sl = []
        self.rewards = []
        self.alpha = alpha
        self.weight = np.zeros(1)

        self.step = 0

        # maximum steps for updating mean_energy. After this number of states we stop updating self.mean_energy
        self.step_energy_limit = 10000

        if 'theta' in self.reg_params:
            if self.reg_params['theta'] != '':
                self.theta = self.reg_params['theta']

        # create and fill DB with theta if it does not exist
        random_db_name = False
        if 'database' in params_db:
            if params_db['database'] == '':
                random_db_name = True
        else:
            random_db_name = True

        if random_db_name:
            letters = string.ascii_lowercase
            params_db['database'] = 'reinforce_' + ''.join(random.choice(letters) for i in range(5))

        self.params_db = params_db
        self.theta_table = theta_table
        self.reinforce_table = reinforce_table
        self.init_sql_queries()

        self.fill_params_db()
        # print(self.params_db)

    def init_sql_queries(self):

        self.sql_select_theta = 'SELECT theta FROM ' + self.theta_table + ' ORDER BY id DESC LIMIT 1'
        self.sql_select_theta_history = 'SELECT theta FROM ' + self.theta_table
        # self.sql_select_reinforce_params = 'SELECT step, " + features + " FROM ' + self.reinforce_table

        # print(self.sql_update_reinforce)
        self.sql_select_energy_history = 'SELECT new_energy FROM ' + self.reinforce_table + ' LIMIT '

        if self.reg_params['reward_limit_last']:
            self.sql_select_reward_history = 'SELECT unscaled_reward FROM ' + self.theta_table + ' ORDER BY id DESC LIMIT '
        else:
            self.sql_select_reward_history = 'SELECT unscaled_reward FROM ' + self.theta_table + ' LIMIT '

    def sql_update_theta(self, unscaled_reward=0, scaled_reward=0):

        if self.reg_params['scale_reward']:
            return f"""INSERT INTO {self.theta_table}(theta, reinforce_id, unscaled_reward, scaled_reward) 
                    VALUES('{json.dumps(self.theta)}', {self.id}, {unscaled_reward}, {scaled_reward})"""

        return f'INSERT INTO {self.theta_table}(theta, reinforce_id) VALUES("{json.dumps(self.theta)}", {self.id})'

    def sql_update_reinforce(self, old_state, new_state, action, alpha):

        f_tablenames, values = self.features_to_tablenames()
        separator = ', '
        features = separator.join(f_tablenames)
        for i in range(len(values)):
            values[i] = '' if values[i] is None else str(values[i])
        values = separator.join(values)
        json_old_struct = old_state.struct_to_str()
        json_new_struct = new_state.struct_to_str()
        return f'''INSERT INTO {self.reinforce_table}(step,
                                                    old_energy, 
                                                    new_energy, 
                                                    action, 
                                                    step_size,
                                                    old_struct,
                                                    new_struct,
                                                    {features}) VALUES 
                                                    ({self.step},
                                                    {old_state.energy},
                                                    {new_state.energy},
                                                    "{action}",
                                                    {alpha},
                                                    "{json_old_struct}",
                                                    "{json_new_struct}", 
                                                    {values})'''

    def features_to_tablenames(self):

        tablenames = []
        values = []

        features = sorted([*self.f_array])

        for f in features:
            if f in self.f_max.keys():
                tablenames.append('max_' + f)
                values.append(self.f_max[f])
            if f in self.f_min.keys():
                tablenames.append('min_' + f)
                values.append(self.f_min[f])
            if f in self.f_mean.keys():
                tablenames.append('mean_' + f)
                if self.f_mean[f] is not None:
                    values.append(self.f_mean[f])
            if f in self.f_std.keys():
                tablenames.append('std_' + f)
                values.append(self.f_std[f])

        return tablenames, values

    # select action according to given energy and current policy
    def select_action(self, state, excluded_actions=[]):

        self.load_params()

        action_space = np.array(self.actions)
        permitted_actions = [x for x in self.actions if x not in excluded_actions]
        # if the list of excluded actions contains all actions we permit all actions
        if len(permitted_actions) == 0:
            permitted_actions = self.actions
            excluded_actions = []

        # first return randomly chosen action if state is not valid (energy is > max)
        # or with epsilon probability
        if state.energy > self.max_energy or self.reg_params['epsilon'] > random.random():
            action_space = np.array(permitted_actions)
            return np.random.choice(action_space)

        # if energy is <= max, then calculate the probabilities and select an action
        action_probs = [0 for k in self.actions]

        for i in range(len(action_probs)):
            action_probs[i] = self.action_prob(self.actions[i], state, excluded_actions)

        # print(f'Old action prob: {old_action_probs}')
        # print(f'New action prob: {action_probs}')

        # max_prob = max(action_probs)
        #
        # if max_prob > self.max_prob:
        #     max_i = action_probs.index(max_prob)
        #
        #     action_probs[max_i] = self.max_prob
        #     for i in range(len(action_probs)):
        #         if i != max_i:
        #             action_probs[i] = action_probs[i] * (1 - self.max_prob) / (1 - max_prob)

        try:
            action = np.random.choice(action_space,
                                      p=action_probs)
        except ValueError as e:
            print(f'Probabilities {action_probs} with error: {e}')
            action = np.random.choice(action_space)

        return action

        # print('Features for action ' + action + ': ' + str(self.features(state, action)))

    # update features scaling parameters: mean, std, min, max
    def calc_f_scaling(self):

        features = [*self.f_array]
        for f in features:
            if f is Reinforce.f_energy or f is Reinforce.f_old_energy:
                # if len(self.f_array[f]) == 0:
                if len(self.f_array[f]) > 1:
                    self.f_mean[f] = statistics.mean(self.f_array[f])
                    self.f_std[f] = np.std(self.f_array[f])
                    self.f_max[f] = max(self.f_array[f])
                    self.f_min[f] = min(self.f_array[f])

    # update features scaling parameters: mean, std, min, max
    def update_f_scaling(self, old_state, new_state, action, alpha):

        if self.step < self.step_energy_limit:
            features = [*self.f_array]
            for f in features:
                if f is Reinforce.f_energy or f is Reinforce.f_old_energy:
                    # if len(self.f_array[f]) == 0:
                    if self.f_mean[f] is None:
                        print('update_f_scaling: Mean is None')
                        self.f_array[f] = [old_state.energy]

                    if new_state.energy < self.max_energy:
                        self.f_array[f].append(new_state.energy)

            self.calc_f_scaling()

        if new_state.energy < self.max_energy:
            self.sql_execute(self.sql_update_reinforce(old_state, new_state, action, alpha))

    # update policy after action that led from old state to new state
    def update(self, action, old_state, new_state, end_episode=False):

        # if the energy of the structure is valid add it to the episode
        if old_state.energy <= self.max_energy:
            if new_state.energy <= self.max_energy or self.non_converge_penalty != 0:
                self.episode.append((old_state, new_state, action))

        # if the episode has the fixed length and achieves it, we end the episode
        # if the episode has the flexible length and we force to end it, end it
        if (len(self.episode) >= self.episode_length and self.fixed_episodes) \
                or (end_episode is True and not self.fixed_episodes):
            return self.end_episode()

    # end the current episode updateing all parameters
    def end_episode(self):

        reward = 0

        # update the policy for each step of episode
        for step in range(len(self.episode)):

            old_state, new_state, action = self.episode[step]

            reward = self.episodic_reward(step, log=True)
            # print(old_state)
            # print(new_state)
            # print(reward)

            pi = self.action_prob(action=action, state=old_state)
            if pi == 0:
                print("policy is zero. do not update policy")
                print("Action is %s, old energy is %f, new energy is %f" % (action,
                                                                            old_state.energy,
                                                                            new_state.energy))
                return reward

            self.load_params()

            unscaled_reward = reward
            if self.reg_params['scale_reward'] and self.std_reward is not None and self.mean_reward is not None:
                reward = (unscaled_reward - self.mean_reward) / self.std_reward
                print('Reward standardization: ' + str(unscaled_reward) + ' --> ' + str(reward))

            if 'max_reward' in self.reg_params:
                reward = min(reward, self.reg_params['max_reward'])
            if 'min_reward' in self.reg_params:
                reward = max(reward, self.reg_params['max_reward'])

            if self.alpha != 0:
                alpha = self.alpha
            else:
                alpha = self.alpha_func(action, old_state)
                # print(alpha)

            self.update_f_scaling(old_state, new_state, action, alpha)
            self.step += 1

            if self.debug:
                print('Reward: ' + str(reward))

            # on the first steps we force reward to be from -1 to +1
            # because features can be standardized using mean and std of energy by too few samples
            if self.step < 10:
                reward = 1 if reward > 1 else reward
                reward = -1 if reward < -1 else reward
                print('Reward forced to -1+1: ' + str(reward))

            if self.debug:
                print('Alpha: ' + str(alpha))
                print('Old theta: ' + str(self.theta))

            diff = ''
            for i in range(len(self.theta)):
                diff_action_prob = self.diff_action_prob(action=action, state=old_state, diff_var=i)
                diff += str(diff_action_prob) + ', '

                if diff_action_prob != 0:
                    entr = self.diff_entropy(action=action, state=old_state, diff_var=i)
                    # if diff_action_prob < 0 and i % 2 == 0:
                    #     beta = self.reg_params['beta']
                    # else:
                    #     beta = - self.reg_params['beta']
                    if self.step > 10:
                        if self.debug:
                            print(f'Diff var is {i}')
                            print(f'Whole update: alpha * (reward * diff_action_prob - beta * entr) / pi')
                            print(f'Whole update: {alpha} * '
                                  f'({reward} * {diff_action_prob} + {self.reg_params["beta"]} * {entr} ) / {pi}')

                        # self.theta[i] += alpha * (reward * diff_action_prob - beta * entr) / pi
                        # do not update reinforce the first 10 steps due to the big mean/std features errors

                        self.theta[i] += alpha * (reward * diff_action_prob + self.reg_params['beta'] * entr) / pi

            if self.debug:
                print('reward: ' + str(reward))
                print('beta: ' + str(self.reg_params['beta']))
                print('Diff: ' + diff)
                print('Probability: ' + str(pi))
                print('New theta: ' + str(self.theta))

            # difference = []
            # zip_object = zip(self.theta, old_theta)
            # for list1_i, list2_i in zip_object:
            #     difference.append(list1_i - list2_i)
            #
            # print(difference)

            # update theta param in DB
            self.sql_execute(self.sql_update_theta(unscaled_reward=unscaled_reward, scaled_reward=reward))
            self.thetas.append(self.theta.copy())

        # refresh episode
        self.episode = []
        return reward

    def diff_entropy(self, state, action, diff_var):

        probs = list(map(self.action_prob, self.actions, [state for i in self.actions]))
        entropy = scipy.stats.entropy(probs)
        action_prob = self.action_prob(action=action, state=state)
        diff_action_prob = self.diff_action_prob(action, state, diff_var)

        if self.debug:
            if diff_action_prob != 0:
                print(f'New entropy diff:')
                print(f'Formula: -diff_action_prob * (math.log(action_prob) + 1) ')
                print(f'Formula: -{diff_action_prob} * ({math.log(action_prob)} + 1) ')
                print(f'diff_entropy is {-diff_action_prob * (math.log(action_prob) + 1)}')

        if entropy < self.reg_params['e_threshold']:
            try:
                return -diff_action_prob * (math.log(action_prob) + 1)
            except ValueError as e:
                e_text = f'{e}: state={str(state)} action={action} diff_var={diff_var} action_prob={action_prob} ' \
                         f'diff_action_prob={diff_action_prob}'
                print(e_text)
                raise
        else:
            return 0

    # reward function depending on the old and new structure parameters
    def reward(self, old_state, new_state, log=False):

        if self.debug:
            print('Old state energy: ' + str(old_state.energy))
            print('New state energy: ' + str(new_state.energy))

        if self.reg_params['smart_penalty']:
            if old_state.energy == new_state.energy or new_state.energy > self.max_energy or not new_state.isunique:
                energies_num = max(1, int(len(self.f_array[Reinforce.f_energy]) / 10))
                high_energy = statistics.mean(heapq.nlargest(energies_num, self.f_array[Reinforce.f_energy]))
                fake_state = new_state.copy()
                fake_state.energy = high_energy
                res = self.feature(Reinforce.f_energy, old_state, log=False) \
                      - self.feature(Reinforce.f_energy, fake_state, log=False)
                print(f'High energy {fake_state.energy} ')
                print(f'Penalty is {res} ')
                return res

        if old_state.energy == new_state.energy:
            res = self.zero_reward_penalty

        elif new_state.energy > self.max_energy:
            res = self.non_converge_penalty

        else:
            if self.reward_type is Reinforce.unscaled:
                res = (old_state.energy - new_state.energy)

            elif self.reward_type is Reinforce.change_in_features:
                res = self.feature(Reinforce.f_energy, old_state, log=log) - self.feature(Reinforce.f_energy, new_state, log=log)
                # print('Reward: ' + str(res))

            elif self.reward_type is Reinforce.uniqueness:
                res = 0
                if new_state.isunique:
                    res -= self.non_unique_penalty

            if self.non_unique_penalty != 0 and not new_state.isunique:
                if self.reg_params['non_unique_reward']:
                    res += self.non_unique_penalty
                else:
                    res = self.non_unique_penalty
                print(f'Penalty for non-uniqueness, reward: {res}')

        # # self.rewards_change.append(self.feature(Reinforce.f_energy, old_state, log=log) - self.feature(Reinforce.f_energy, new_state, log=log))
        # self.rewards_change.append(self.f_std[Reinforce.f_energy])
        # # if abs(self.rewards_change[-1]) > 10:
        # #     print("TOO BIG REWARD: " + str(self.rewards_change[-1]))
        # #     time.sleep(10)
        # tmp = self.feature(Reinforce.f_energy, old_state, log=log) - self.feature(Reinforce.f_energy, new_state, log=log)
        # if self.mean_reward is not None and self.std_reward != 0 and self.std_reward is not None:
        #     tmp = (tmp - self.mean_reward) / self.std_reward
        # self.rewards_standard_change.append(tmp)
        # if abs(self.rewards_standard_change[-1]) > 10:
        #     print("TOO BIG REWARD: " + str(self.rewards_standard_change[-1]))
        #     time.sleep(10)

        return res

    # reward function depending on the old and new structure parameters
    def episodic_reward(self, step, log=False):

        res = 0
        discount = 1

        for i in range(step, len(self.episode)):
            old_state, new_state, action = self.episode[i]
            res += (discount ** i) * self.reward(old_state, new_state, log=log)

        # print(f'Episodic reward is {res}')
        return res

    # standardize feature by formula: (value - mean) / std
    def standardize_feature(self, f, val, default_val=0, log=False):

        if log and self.debug:
            print('Feature standardisation')
        if self.f_mean[f] is not None and self.f_std[f] != 0 and self.f_std[f] is not None:
            if log and self.debug:
                print('Mean energy: ' + str(self.f_mean[f]))
                print('Std energy: ' + str(self.f_std[f]))
                print('Current energy: ' + str(val))
                print('Feature value: ' + str((val - self.f_mean[f]) / self.f_std[f]))
            return (val - self.f_mean[f]) / self.f_std[f]
        else:
            if log and self.debug:
                print('Default energy: ' + str(default_val))
            return default_val

    # normalize feature to be from min to max
    def normalize_feature(self, f, val, default_val=0, min_val=0, max_val=1):

        if self.f_max[f] is not None and self.f_min[f] is not None:
            return min_val + (val - self.f_min[f]) * (max_val - min_val) / (self.f_max[f] - self.f_min[f])
        else:
            return default_val

    def feature(self, f, state, log=False):

        if f is Reinforce.f_energy:
            return self.standardize_feature(f, state.energy, default_val=state.energy, log=log)

        if f is Reinforce.f_old_energy:
            return self.standardize_feature(f, state.old_energy, default_val=state.old_energy, log=log)

        if f is Reinforce.f_sl:
            return self.normalize_feature(f, state.sl(), default_val=state.sl(), min_val=-1, max_val=-1)

        return None

    # returns features vector for given state and action
    # the vector size is number of actions * number of features
    def features(self, state, action):

        features = sorted(self.f_array.keys())

        res = np.zeros(len(features)*len(self.actions))

        f_counter = 0

        # fill energy features
        if Reinforce.f_energy in features:
            f = Reinforce.f_energy
            res[len(features) * self.actions.index(action) + f_counter] = self.standardize_feature(f, state.energy)
            f_counter += 1

        # fill old energy features
        if Reinforce.f_old_energy in features:
            f = Reinforce.f_old_energy
            res[len(features) * self.actions.index(action) + f_counter] = self.standardize_feature(f, state.old_energy)
            f_counter += 1

        # fill stack length features
        if Reinforce.f_sl in features:
            f = Reinforce.f_sl
            res[len(features) * self.actions.index(action) + f_counter] = self.normalize_feature(f, state.sl(),
                                                                                                 default_val=0,
                                                                                                 min_val=-1,
                                                                                                 max_val=+1)
            f_counter += 1

        return res

    # probability to be chosen for given action in given state, softmax function
    def old_action_prob(self, action, state, excluded_actions=[]):

        if action in excluded_actions:
            return 0

        exp_sum = 0

        try:
            exp = math.exp(self.beta * self.h(state, action))
            exp_sum = self.exp_sum(state, excluded_actions)

            if exp_sum > 0:
                return exp / exp_sum
            else:
                print(f"EXP sum is zero: action is {action}, energy is {state.energy}, exp is {exp}, exp_sum is {exp_sum}")

        except OverflowError as e:
            print(e)
            raise Exception(f'''action_prob error: step is {self.step}, h is {self.h(state, action)}, action is {action}, \nstate is {state},
                    Features are {self.features(state, action)}, \ntheta is {self.theta}''')

        return 0

    def action_prob(self, action, state, excluded_actions=[]):

        if action in excluded_actions:
            return 0

        try:
            h_list = []
            h = 0
            for a in self.actions:
                if a not in excluded_actions:
                    h_list.append(self.beta * self.h(state, a))
                    if a == action:
                        h = h_list[-1]

            h_list = np.array(h_list)
            max_h = h_list.max()
            h -= max_h
            h_list -= max_h

            return math.exp(h) / np.exp(h_list).sum()

        except OverflowError as e:
            raise Exception(f'''action_prob error: {e}, step is {self.step}, h is {self.h(state, action)}, action is {action}, \nstate is {state},
                    Features are {self.features(state, action)}, \ntheta is {self.theta}''')

        return 0

    @staticmethod
    def st_action_prob(action, state, theta, actions,
                       f_mean, f_std, f_max, f_min, f_array,
                       excluded_actions=[]):

        # print('Action: ' + action)
        # print('State: ' + str(state))
        # print('Theta: ' + str(theta))
        # print('Actions: ' + str(actions))
        # print('mean_energy: ' + str(mean_energy))
        reinforce = Reinforce(actions=actions, theta=theta, features_set=f_array.keys())
        reinforce.f_array = f_array
        reinforce.f_mean = f_mean
        reinforce.f_std = f_std
        reinforce.f_min = f_min
        reinforce.f_max = f_max
        return reinforce.action_prob(action, state, excluded_actions)

    # sum of exponents for softmax function
    def exp_sum(self, state, excluded_actions=[]):

        res = 0
        for i in range(len(self.actions)):
            if not self.actions[i] in excluded_actions:
                # print(self.h(state, self.actions[i]))
                res += math.exp(self.beta * self.h(state, self.actions[i]))

        return res

    # derivative of policy on diff_var param
    def diff_action_prob(self, action, state, diff_var):

        # exp_sum = -1
        # exp_h = -1
        diff_h = -1

        try:
            # exp_sum = self.exp_sum(state)
            # exp_h = math.exp(self.beta * self.h(state, action))
            diff_h = self.diff_h(state, action, diff_var)

            action_prob = self.action_prob(action, state)

            # print(f'Old diff: {(exp_h * diff_h * (exp_sum - exp_h) / (exp_sum ** 2))}')
            # print(f'New diff: {(action_prob * diff_h * (1 - action_prob))}')
            # return exp_h * diff_h * (exp_sum - exp_h) / (exp_sum ** 2)

            return action_prob * diff_h * (1 - action_prob)

        except OverflowError as e:
            print(f'diff_action_prob function error: {e}, diff_h is {diff_h}')
            raise Exception(f'''diff_action_prob function error: {e}, 
                                diff_h is {diff_h},
                                step is {self.step}, h is {self.h(state, action)}, 
                                action is {action}, \nstate is {state},
                                Features are {self.features(state, action)}, \ntheta is {self.theta}''')
        except ZeroDivisionError as e:
            raise Exception(f'''diff_action_prob function error: {e}, 
                                diff_h is {diff_h},
                                step is {self.step}, h is {self.h(state, action)}, 
                                action is {action}, \nstate is {state},
                                Features are {self.features(state, action)}, \ntheta is {self.theta}''')


    # derivative of policy as a vector
    def diff_action_prob_vector(self, action, state):

        diff_vector = []

        for diff_var in range(len(self.theta)):
            diff_vector.append(self.diff_action_prob(action, state, diff_var))

        return diff_vector

    @staticmethod
    def sigmoid(x):
        # return np.sign(x) * (np.abs(x)) ** (1 / 3)
        # return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        # return math.atan(x)
        return x

    @staticmethod
    def sigmoid_diff(q, x):
        # return (np.sign(x) * (np.abs(x)) ** (1 / 3))
        # sigmoid = Reinforce.sigmoid(q * x)
        # return (1 - sigmoid ** 2) * x
        # return math.atan(x)
        return x

    # function for action preferences
    def h(self, state, action):

        features = self.features(state, action)

        start_coef = []

        for i in range(len(features)):
            if features[i] != 0:
                start_coef.append(self.theta[2*i + 1])
                if self.h_type != 'linear':
                    features[i] = np.sign(features[i]) * (np.abs(features[i])) ** (1 / 3)

        # print(f'h: sum theta: {sum(self.theta[1::2])}')
        # print(f'theta 0 is: {self.theta[1::2]}')
        # print(f'h: sum theta: {np.multiply(self.theta[::2], features).sum()}')
        # print(f'theta 1 is: {self.theta[::2]}')

        return np.multiply(self.theta[::2], features).sum() + sum(start_coef) * self.free_term
        # return sum(start_coef) * self.free_term

    # derivative of h is either energy if action corresponds to diff_var or zero
    def diff_h(self, state, action, diff_var):

        f_num = len(self.f_array.keys())

        if f_num * self.actions.index(action) * 2 <= diff_var < f_num * (self.actions.index(action) + 1) * 2:
            # diff for theta * energy^(1/3)
            if diff_var % 2 == 0:
                features = self.features(state, action)
                if self.h_type == 'linear':
                    return features[int(diff_var/2)]
                else:
                    return np.sign(features[int(diff_var / 2)]) * (np.abs(features[int(diff_var / 2)])) ** (1 / 3)
            else:
                # diff for theta
                return self.free_term
        else:
            return 0

    # power (1/3) function for action preferences
    # def h(self, state, action):
    #
    #     features = self.features(state, action)
    #
    #     start_coef = []
    #
    #     for i in range(len(features)):
    #         if features[i] != 0:
    #             start_coef.append(self.theta[2*i + 1])
    #             features[i] = np.sign(features[i]) * (np.abs(features[i])) ** (1 / 3)
    #
    #     # print(f'theta 0 is: {np.multiply(self.theta[::2], features).sum()}')
    #     # print(f'theta 1 is: {sum(start_coef) * 0.1}')
    #
    #     return np.multiply(self.theta[::2], features).sum() + sum(start_coef) * self.free_term

    # derivative of h is either energy if action corresponds to diff_var or zero
    # def diff_h(self, state, action, diff_var):
    #
    #     f_num = len(self.f_array.keys())
    #
    #     if f_num * self.actions.index(action) * 2 <= diff_var < f_num * (self.actions.index(action) + 1) * 2:
    #         # diff for theta * energy^(1/3)
    #         if diff_var % 2 == 0:
    #             features = self.features(state, action)
    #             return np.sign(features[int(diff_var/2)]) * (np.abs(features[int(diff_var/2)])) ** (1 / 3)
    #         else:
    #             # diff for theta
    #             return self.free_term
    #     else:
    #         return 0

    def armijo_cond(self, t, action, state):

        diff_action_prob_vector = self.diff_action_prob_vector(action, state)

        x = np.array(diff_action_prob_vector)
        vector_length = np.linalg.norm(x) ** 2

        d_action_prob = Reinforce.st_action_prob(action,
                                                 state,
                                                 [sum(x) for x in zip(self.theta, diff_action_prob_vector)],
                                                 actions=self.actions,
                                                 mean_energy=self.mean_energy,
                                                 std_energy=self.std_energy)

        action_prob = self.action_prob(action, state)

        # print('delta f length ' + str(vector_length))
        # print('delta f ' + str(diff_action_prob_vector))
        # # print('theta ' + str(self.theta))
        # print('f(x + delta f) ' + str(d_action_prob))
        # print('f + t/2*(delta f)^2 ' + str(action_prob + t / 2 * vector_length))
        # print('Result is ' + str(d_action_prob >= action_prob + t / 2 * vector_length))

        return d_action_prob >= action_prob + t / 2 * vector_length

    def alpha_func(self, action, state, beta=0.8):

        x = 5000

        t = (x)/(100*x + self.step)
        print(t)
        return t

        # while not self.armijo_cond(t, action, state):
        #     t *= beta
        #     # print('New t is ' + str(t))
        #     # if (t < 0.0000001):
        #     #     print('Too small t')
        #     #     break
        #
        # print('alpha is: ' + str(t))
        # return t

    def state_value(self, state):

        return self.feature(Reinforce.f_energy, state) * self.weight[0]

    def load_params(self):

        row = self.sql_execute(self.sql_select_theta, result='one')
        # print(self.sql_select_theta)
        if row is not None:
            self.theta = json.loads(row[0])
        else:
            print(f'Reading theta from DB is failed')

        # load energy params
        # print(self.sql_select_energy_history + str(self.step_energy_limit))
        if Reinforce.f_energy in self.f_array.keys():
            rows = self.sql_execute(self.sql_select_energy_history + str(self.step_energy_limit),
                                    result='all')
            if rows is not None:
                energies = [item[0] for item in rows]
                if len(energies) > 0:
                    self.f_array[Reinforce.f_energy] = energies
                    if Reinforce.f_old_energy in self.f_array.keys():
                        self.f_array[Reinforce.f_old_energy] = energies
            else:
                print(f'Reading energies from DB is failed')

        self.calc_f_scaling()

        if self.reg_params['scale_reward']:
            if 'step_reward_limit' in self.reg_params:
                limit = self.reg_params['step_reward_limit']
            else:
                limit = self.step_energy_limit
            rows = self.sql_execute(self.sql_select_reward_history + str(limit),
                                    result='all')
            if rows is not None:
                rewards = [item[0] for item in rows]
                if len(rewards) > 1:
                    self.mean_reward = statistics.mean(rewards)
                    self.std_reward = np.std(rewards)
                else:
                    self.mean_reward = None
                    self.std_reward = None
            else:
                self.mean_reward = None
                self.std_reward = None
                print(f'Reading rewards from DB is failed')


    def print_probabilities(self, state, excluded_actions=[]):

        draws = {k: 0 for k in self.actions}

        for k, v in draws.items():
            draws[k] = round(self.action_prob(k, state=state, excluded_actions=excluded_actions), 3)

        print("For energy %f the probabilities are" % state.energy)
        print(draws)

    def draw_actions_probabilities(self, min_energy, max_energy, step=0.001,
                                   filename="reinforce_actions_probabilities.png",
                                   title='Actions preferences',
                                   excluded_actions=[],
                                   max_y=100,
                                   small=False):

        plt.figure(figsize=(800 / 150, 600 / 150), dpi=150)

        if small:
            fontsize = 24
            tickssize = 18
            ratio = 1
            plt.locator_params(nbins=4)
            linewidth = 3
        else:
            fontsize = 13
            tickssize = 13
            ratio = 6/8
            plt.locator_params(nbins=6)
            linewidth = 2.5

        x = np.arange(min_energy, max_energy, step)

        plt.rcParams['font.family'] = "sans-serif"
        plt.rcParams['font.sans-serif'] = "Arial"

        for action in self.actions:
            if action not in excluded_actions:
                y = []
                for i in x:
                    state = State(i)
                    prob = self.action_prob(action=action, state=state, excluded_actions=excluded_actions)
                    prob = prob * 100 * (1 - self.reg_params['epsilon']) \
                           + 1 / len(self.actions) * 100 * self.reg_params['epsilon']
                    y.append(prob)

                line, = plt.plot(x, y, linewidth=linewidth)
                line.set_label(action)
                # plt.legend(prop={'size': 13}, loc='upper right', frameon=False)

        axes = plt.gca()
        axes.set_ylim([0, max_y])
        # if not small:
        plt.legend(prop={'size': 14}, loc='upper right', frameon=False)

        plt.title(title, fontsize=fontsize)
        plt.xlabel('Energy (eV/atom)', fontsize=tickssize)
        plt.ylabel('Probability (%)', fontsize=tickssize)
        plt.xticks(fontsize=tickssize)
        plt.yticks(fontsize=tickssize)

        x_left, x_right = plt.xlim()
        y_low, y_high = plt.ylim()
        plt.gca().set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

        plt.savefig(filename,
                    bbox_inches='tight',
                    pad_inches=0.1)
        plt.clf()

    def entropy(self, state):

        res = 0

        for a in self.actions:
            prob = self.action_prob(a, state)
            if prob == 0:
                prob = 0.00001
            try:
                res += prob * math.log(prob)
            except ValueError as e:
                print(e)

        return -res

    def animated_actions_probabilities(self, min_energy, max_energy, filename, interval=10,
                                       sl=[], old_energy=0, max_y=100):

        # if there is no db we work with then we don't need to load theta
        rows = self.sql_execute(self.sql_select_theta_history, result='all')
        thetas = []
        for row in rows:
            thetas.append(json.loads(row[0]))

        if old_energy == 0:
            old_energy = min_energy

        # General plot parameters
        mpl.rcParams['font.family'] = 'Avenir'
        mpl.rcParams['font.size'] = 16
        mpl.rcParams['axes.linewidth'] = 2
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['xtick.major.size'] = 10
        mpl.rcParams['xtick.major.width'] = 2
        mpl.rcParams['ytick.major.size'] = 10
        mpl.rcParams['ytick.major.width'] = 2
        mpl.rcParams["legend.loc"] = 'center right'

        actions_num = len(self.actions)

        fig, ax = plt.subplots()

        if len(sl) > 0:
            lines = [0] * (actions_num * len(sl))
        else:
            lines = [0] * (actions_num + 1)

        for i in range(len(lines)):
            lines[i], = plt.plot([min_energy, max_energy], [0, max_y])

        T = np.linspace(1, len(thetas), int(len(thetas)/interval))

        temp = ax.text(int((min_energy + max_energy)/2), 100, '', ha='right', va='top', fontsize=20)

        plt.title("Actions probability distribution")
        plt.xlabel("Energy (eV/atom)")
        plt.ylabel("Probability (%)")

        # colors = plt.get_cmap('gist_ncar', len(lines) + 3)
        colors = plt.get_cmap('tab10', len(lines) + 3)

        remember_theta = self.theta

        if len(sl) > 0:
            # Animation function
            def animate(i):
                x = np.linspace(min_energy, max_energy, 100)
                for local_sl in sl:
                    for a in range(actions_num):
                        y = np.ones(len(x))
                        for j in range(len(x)):
                            self.theta = thetas[int(T[i]-1)]
                            y[j] = 100 * self.action_prob(self.actions[a],
                                                          State(x[j], np.zeros(local_sl), old_energy=old_energy))
                        lines[len(sl)*a + sl.index(local_sl)].set_data(x, y)
                        lines[len(sl)*a + sl.index(local_sl)].set_label(self.actions[a] + ' sl=' + str(local_sl))
                        lines[len(sl) * a + sl.index(local_sl)].set_color(
                                colors(len(sl) * a + sl.index(local_sl)))

                plt.legend()

                temp.set_text(str(int(T[i])) + ' steps')
                # temp.set_color(colors(i))
        else:
            # Animation function
            def animate(i):
                x = np.linspace(min_energy, max_energy, 100)
                for a in range(actions_num):
                    y = np.ones(len(x))
                    for j in range(len(x)):
                        self.theta = thetas[int(T[i] - 1)]
                        y[j] = 100 * self.action_prob(self.actions[a],
                                                      State(x[j], old_energy=old_energy))
                    lines[a].set_data(x, y)
                    lines[a].set_label(self.actions[a])
                    lines[a].set_color(colors(a))
                # entropy
                y = np.ones(len(x))
                for j in range(len(x)):
                    self.theta = thetas[int(T[i] - 1)]
                    y[j] = 10 * self.entropy(State(x[j], old_energy=old_energy))
                lines[actions_num].set_data(x, y)
                lines[actions_num].set_label('Entropy')
                lines[actions_num].set_color(colors(actions_num))

                plt.legend()

                temp.set_text(str(int(T[i])) + ' steps')
                # temp.set_color(colors(i))

        # Create animation
        ani = FuncAnimation(fig=fig, func=animate, frames=range(len(T)), interval=50, repeat=True)

        # Ensure the entire plot is visible
        fig.tight_layout()

        # Save and show animation
        ani.save(filename, writer=PillowWriter(fps=2))

        self.theta = remember_theta

        # plt.show()

    # creates and fills the theta_params table with predefined theta
    def fill_params_db(self):

        # create DB
        sql_create_db = f'CREATE DATABASE IF NOT EXISTS {self.params_db["database"]}'
        self.sql_execute(sql_create_db, create_db=True)

        # create reinforce table
        separator = ' double, '
        features = separator.join(self.features_to_tablenames()[0]) + ' double'
        sql_create_reinforce_table = "CREATE TABLE IF NOT EXISTS " + self.reinforce_table + """ (
                                            id INT AUTO_INCREMENT PRIMARY KEY,
                                            step int,
                                            old_energy double,
                                            new_energy double,
                                            action varchar(10),
                                            step_size double,
                                            old_struct varchar(200),
                                            new_struct varchar(200),""" \
                                     + features + "); "

        print(sql_create_reinforce_table)

        self.sql_execute(sql_create_reinforce_table)

        reward_str = ''
        if self.reg_params['scale_reward']:
            reward_str = ', unscaled_reward double, scaled_reward double'
        # create theta table
        sql_create_theta_table = f"""CREATE TABLE IF NOT EXISTS {self.theta_table} (
                                            id INT AUTO_INCREMENT PRIMARY KEY,
                                            theta text,
                                            reinforce_id tinyint {reward_str}
                                        ); """

        self.sql_execute(sql_create_theta_table)

        # if the theta table is empty fill it with default value
        row = self.sql_execute(f'SELECT COUNT(*) FROM {self.theta_table}', result='one')
        if row is None:
            print(f'Reading the number of thetas in DB is failed')
        else:
            if row[0] < 1:
                self.sql_execute(self.sql_update_theta())

    # open connection to db, execute the query in <max_attempts> attemps, close connection
    # result parameter can be 'one', 'all', if we do the select statement, or 'none' if the query is update
    def sql_execute(self, sql, max_attempts=5, result='none', create_db=False):

        failed = True
        attempts = 0

        while attempts < max_attempts and failed:

            try:
                # conn_db = sqlite3.connect(self.params_db)
                conn_db = connect(
                    host=self.params_db['host'],
                    user=self.params_db['user'],
                    password=self.params_db['password'],
                    database='' if create_db else self.params_db['database'],
                    auth_plugin='mysql_native_password'
                )
                try:
                    cur = conn_db.cursor(buffered=(result != 'none'))
                    cur.execute(sql)
                    conn_db.commit()
                    failed = False
                    if self.debug:
                        print(f'DB query {sql}: success')

                    if result == 'one':
                        return cur.fetchone()
                    elif result == 'all':
                        return cur.fetchall()
                    cur.close()
                except Error as e:
                    attempts += 1
                    print(f'DB query {sql} failed in attempt {attempts} with error: {e}')
                    time.sleep(10)

                conn_db.close()

            except Error as e:
                attempts += 1
                print(f'DB query {sql} failed in attempt {attempts} with error: {e}')
                time.sleep(10)