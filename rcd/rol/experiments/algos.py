import numpy as np
from garage import wrap_experiment
from garage.envs import GraphEnv
from garage.experiment import Snapshotter
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from garage.torch.algos import SGDH
from garage.torch.optimizers import OptimizerWrapper
from garage.torch.policies import ArgMaxMLPPolicy
from garage.trainer import Trainer
from itertools import combinations

import torch
from graph_utils import get_stat, load_graph_data, load_partial_graph_data


def define_algorithm(alg_name, experiment_name, n_epochs=None, sampler_batch_size=None, alpha=None, data_size=None,
                     lr=None, oracle=None, partial=False, erdos=False, log_path=None, graph_path=None):
    if alg_name == 'vpg':
        @wrap_experiment(archive_launch_repo=False,
                         log_dir=log_path + "/" + experiment_name)
        def sl_vpg(ctxt=None, seed=0, ):
            """Train PPO with InvertedDoublePendulum-v2 environment.

            Args:
                ctxt (garage.experiment.ExperimentContext): The experiment
                    configuration used by Trainer to create the snapshotter.
                seed (int): Used to seed the random number generator to produce
                    determinism.

            """

            set_seed(seed)
            if partial:
                data, A = load_partial_graph_data(graph_path, data_size, seed)
            else:
                data, A = load_graph_data(graph_path, data_size, seed, erdos)

            env = GraphEnv(data=data, alpha=alpha, A=A, oracle=oracle)
            policy = ArgMaxMLPPolicy(env.spec, hidden_sizes=[64, 64],
                                     hidden_nonlinearity=torch.nn.ReLU,
                                     output_nonlinearity=None,
                                     )

            policy_optimizer = OptimizerWrapper((torch.optim.Adam, {"lr": lr}), policy)
            value_function = LinearFeatureBaseline(env_spec=env.spec)
            sampler = RaySampler(agents=policy,
                                 envs=env,
                                 max_episode_length=env.spec.max_episode_length)

            trainer = Trainer(ctxt)
            algo = SGDH(env_spec=env.spec,
                        policy=policy,
                        value_function=value_function,
                        sampler=sampler,
                        discount=1,
                        policy_optimizer=policy_optimizer,
                        neural_baseline=False,
                        )

            trainer.setup(algo, env)
            trainer.train(n_epochs=n_epochs, batch_size=sampler_batch_size)

        def predict(itr='last'):
            snapshotter = Snapshotter()
            data = snapshotter.load(log_path + '/' + experiment_name, itr=itr)
            policy = data['algo'].policy
            env = data['env']
            A = env.adjacency_matrix
            # testing the policy
            print("testing the policy...")
            obs = env.reset()[0]  # The initial observation
            steps = 0

            while steps < env.spec.observation_space.flat_dim:
                env_step = env.step(policy.get_action(obs)[0], test_mode=True)
                obs, rew, = env_step.observation, env_step.reward,
                steps += 1
            generated_graph = np.array(env.generated_graph)
            np.save(log_path + '/vpg_generated_graph_' + experiment_name, generated_graph)
            return get_stat(A=A, B=generated_graph)

        return sl_vpg, predict

    elif alg_name == 'VI':

        def unpackbits(x, num_bits):
            if np.issubdtype(x.dtype, np.floating):
                raise ValueError("numpy data type needs to be int-like")
            xshape = list(x.shape)
            x = x.reshape([-1, 1])
            mask = 2 ** np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
            return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])

        def obs_to_number(array):
            sum_nums = 0
            for i in range(len(array)):
                sum_nums += array[i] * 2 ** i
            return sum_nums

        def generator(num_bits):
            powers = [2 ** exponent for exponent in range(num_bits - 1, -1, -1)]
            for num_on in range(1, num_bits + 1, ):
                for positions in combinations(range(num_bits), num_on):
                    yield sum(powers[position] for position in positions)

        def sl_vi(seed=47, ):
            global V_action
            """
            Args:
                ctxt (garage.experiment.ExperimentContext): The experiment
                    configuration used by Trainer to create the snapshotter.
                seed (int): Used to seed the random number generator to produce
                    determinism.
            """

            set_seed(seed)
            if partial:
                data, A = load_partial_graph_data(graph_path, data_size, seed)
            else:
                data, A = load_graph_data(graph_path, data_size, seed, erdos)

            env = GraphEnv(data=data, alpha=alpha, A=A)
            SMALL_ENOUGH = 1e-7  # threshold to declare convergence
            GAMMA = 1  # discount factor

            V = [-1000] * 2 ** A.shape[1]
            V[0] = 0
            print("number of states: ", len(V))
            V_action = [0] * 2 ** A.shape[1]
            iteration = 0
            while iteration < 1:
                print("VI iteration %d: " % iteration)
                print("\n\n")
                biggest_change = 0
                for s in generator(A.shape[1]):
                    old_v = V[s]
                    state_vector = unpackbits(np.array(s), A.shape[0])
                    new_v = float('-inf')
                    new_a = 0
                    possible_actions = np.where(state_vector == 1)[0]
                    # for each action
                    for a in possible_actions:
                        env.set_state(state_vector)
                        env_step = env.step(a)
                        obs, r, = env_step.observation, env_step.reward,
                        sprime = obs_to_number(obs)
                        v = r + GAMMA * V[sprime]

                        if v > new_v:  # is this the best action so far
                            new_v = v
                            new_a = a
                    # print("***************")
                    V[s] = new_v
                    V_action[s] = new_a
                    # print(V)
                    biggest_change = max(biggest_change, np.abs(old_v - V[s]))

                print('\t biggest change is: %f \n\n' % biggest_change)
                if biggest_change < SMALL_ENOUGH:
                    break
                iteration += 1

        def predict(seed=47, ):
            set_seed(seed)

            if partial:
                data, A = load_partial_graph_data(graph_path, data_size, seed)
            else:
                data, A = load_graph_data(graph_path, data_size, seed, erdos)

            env = GraphEnv(data=data, alpha=alpha, A=A)
            print("testing the policy...")
            obs = env.reset()[0]  # The initial observation
            print(obs)
            obs = int(obs_to_number(obs))
            steps = 0
            policy = []
            while steps < env.spec.observation_space.flat_dim:
                policy.append(V_action[obs])
                env_step = env.step(policy[-1], test_mode=True)
                obs = env_step.observation
                obs = int(obs_to_number(obs))
                steps += 1

            generated_graph = np.array(env.generated_graph)
            np.save(log_path + '/vi_generated_graph_' + experiment_name, generated_graph)
            return get_stat(A=A, B=generated_graph)

        return sl_vi, predict
