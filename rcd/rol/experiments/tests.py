#!/usr/bin/env python3
import numpy as np
from graph_utils import generate_data, get_stat, load_adjacency_matrix, real_markov_boundary
from garage.experiment import Snapshotter
import unittest
from garage.envs import GraphEnv



class TestGraphEnvMethods(unittest.TestCase):

    def test_load_graph(self):
        data, A = generate_data(5000, "data/alarm.mat")
        self.assertEqual(A.shape[0], data.shape[1])

    def test1_get_stat_function(self, ):
        a = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        b = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
        edges_of_A, edges_of_B, extra_edges, missing_edges, \
        precision, recall, skeleton_F1_score = get_stat(a, b)
        self.assertEqual(edges_of_A, 1)
        self.assertEqual(edges_of_B, 2)
        self.assertEqual(extra_edges, 2)
        self.assertEqual(missing_edges, 1)

    def test2_get_stat_function(self, ):
        a = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        b = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        edges_of_A, edges_of_B, extra_edges, missing_edges, \
        precision, recall, skeleton_F1_score = get_stat(a, b)
        self.assertEqual(edges_of_A, 1)
        self.assertEqual(edges_of_B, 3)
        self.assertEqual(extra_edges, 2)
        self.assertEqual(missing_edges, 0)

    def test_initial_env_state(self):
        data, A = generate_data(10000, "data/alarm.mat")
        env = GraphEnv(data=data, alpha=0.001)
        obs = env.reset()[0]
        second = np.ones(env._observation_space.flat_dim)
        np.testing.assert_array_equal(obs, second)

    # def test_generated_graph_edges_equal_to_found_neighbors(self):
    #     obs = env.reset()[0]  # The initial observation
    #     policy.reset()
    #     steps = 0
    #     cumulative_reward = 0
    #     while steps < env.spec.observation_space.flat_dim:
    #         env_step = env.step(policy.get_action(obs)[0], test_mode=True)
    #         obs, rew = env_step.observation, env_step.reward
    #         cumulative_reward += rew
    #         steps += 1
    #     generated_graph = np.array(env.generated_graph)
    #     self.assertLessEqual(np.sum(generated_graph), np.sum(env.first_markov_boundary))
    #     self.assertEqual(np.sum(generated_graph), -cumulative_reward)

    def test_IC_test_neighbor(self):
        data, A = generate_data(10000, "data/alarm.mat")
        env = GraphEnv(data=data, alpha=0.001)
        _ = env.reset()[0]  # The initial observation
        IC_res = env.IC_test(26, 35, [4, 5])
        self.assertEqual(IC_res, False)

    def test_IC_test_neighbor2(self):
        data, A = generate_data(10000, "data/alarm.mat")
        env = GraphEnv(data=data, alpha=0.001)
        _ = env.reset()[0]  # The initial observation
        IC_res = env.IC_test(26, 35, [])
        self.assertEqual(IC_res, False)

    def test_real_markov_boundary(self):
        A = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
        real_mb = real_markov_boundary(A)
        edges_of_A, edges_of_B, extra_edges, missing_edges, \
        precision, recall, skeleton_F1_score = get_stat(A, real_mb)
        self.assertEqual(edges_of_A, 2)
        self.assertEqual(edges_of_B, 3)
        self.assertEqual(missing_edges, 0)
        self.assertEqual(extra_edges, 1)

    def test_real_markov_boundary2(self):
        A = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 0]])
        real_mb = real_markov_boundary(A)
        edges_of_A, edges_of_B, extra_edges, missing_edges, \
        precision, recall, skeleton_F1_score = get_stat(A, real_mb)
        self.assertEqual(edges_of_A, 3)
        self.assertEqual(edges_of_B, 4)
        self.assertEqual(missing_edges, 0)
        self.assertEqual(extra_edges, 1)


    def test_real_markov_boundary3(self):
        data, A = generate_data(5000, "data/alarm.mat")
        real_mb = real_markov_boundary(A)
        edges_of_A, edges_of_B, extra_edges, missing_edges, \
        precision, recall, skeleton_F1_score = get_stat(A, real_mb)
        self.assertEqual(edges_of_A, 46)
        self.assertEqual(missing_edges, 0)

    def test_generated_mb_vs_real(self):
        data, A = generate_data(10000, "data/alarm.mat")
        env = GraphEnv(data=data, alpha=0.001)
        first_mb = env.first_markov_boundary
        real_mb = real_markov_boundary(A)
        edges_of_A, edges_of_B, extra_edges, missing_edges, \
        precision, recall, skeleton_F1_score = get_stat(real_mb, first_mb)
        self.assertGreaterEqual(edges_of_B, 66)
        self.assertEqual(missing_edges, 0)

    def test_Oracle_CI_test_function(self):
        data, A = generate_data(10000, "data/alarm.mat")
        env = GraphEnv(data=data, alpha=0.001, A=A)
        _ = env.reset()[0]  # The initial observation
        IC_res = env.oracle_CI(26, 35, [4, 5])
        self.assertEqual(IC_res, False)

    def test_Oracle_CI_test_function2(self):
        data, A = generate_data(10000, "data/alarm.mat")
        env = GraphEnv(data=data, alpha=0.001, A=A)
        _ = env.reset()[0]  # The initial observation
        IC_res = env.oracle_CI(26, 35, [])
        self.assertEqual(IC_res, False)

    def test_Oracle_CI_test_function3(self):
        data, A = generate_data(10000, "data/alarm.mat")
        env = GraphEnv(data=data, alpha=0.001, A=A)
        _ = env.reset()[0]  # The initial observation
        env.remove_target(0)
        env.remove_target(0)
        IC_res = env.oracle_CI(26, 3, [4, 1])
        # self.assertEqual(IC_res, False)
if __name__ == '__main__':
    unittest.main()
