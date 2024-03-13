#!/usr/bin/env python3
from algos import define_algorithm
import argparse

parser = argparse.ArgumentParser(description='IS-MBPG')
parser.add_argument('--alg_name', default='vpg', help='choose the algorithm from [vpg, vi]')
parser.add_argument('--graph_name', default='sachs',
                    help='choose the graph name in your dataset like [cancer, asia, sachs, alarm, barley, hepar2]')
parser.add_argument('--log_path', default='/root/Data/log_graphs',)
parser.add_argument('--dataset_path', default='/root/Data/sl_data',
                    help='put the dataset path where the graphs are in')
parser.add_argument('--n_epochs', default=100, type=int, )
parser.add_argument('--seed', default=0, type=int, )
parser.add_argument('--data_num', default=10000, type=int, )
parser.add_argument('--sampler_batch_size', default=1000, type=int, )
parser.add_argument('--alpha', default=0.05, type=float, )
parser.add_argument('--lr', default=5e-3, type=float, )
parser.add_argument('--oracle', default=False, type=bool, )
parser.add_argument('--IS_MAG', default=False, type=bool, )

args = parser.parse_args()


def run_task(args):
    graph_path = args.dataset_path + '/' + args.graph_name

    if args.IS_MAG:
        experiment_name = "MAG_graph_discovery_vpg_{}_bs={}alpha={}lr{}oracle={}_seed={}data_num{}".format(
            args.graph_name,
            args.sampler_batch_size,
            args.alpha,
            args.lr,
            args.oracle, args.seed,
            args.data_num)
    else:
        experiment_name = "DAG_graph_discovery_vpg_{}_bs={}alpha={}lr{}oracle={}_seed={}data_num{}".format(
            args.graph_name,
            args.sampler_batch_size,
            args.alpha,
            args.lr,
            args.oracle, args.seed, args.data_num)

    alg, predict = define_algorithm(alg_name=args.alg_name, experiment_name=experiment_name, n_epochs=args.n_epochs,
                           sampler_batch_size=args.sampler_batch_size, alpha=args.alpha,
                           data_size=args.data_num,
                           lr=args.lr, oracle=args.oracle, partial=args.IS_MAG, graph_path=graph_path, log_path=args.log_path)
    alg(seed=args.seed)
    stat = predict(itr="last",)
    print(stat)


run_task(args)
