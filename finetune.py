from functions import Refiner
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=str, default='test_exp', help='checkpoint dir')
    parser.add_argument('-e', '--epoch', type=int, default=100, help='fintune epochs number')
    parser.add_argument('-d', '--dataset', type=str, default='AFLW2000', help='testing dataset (AFLW2000/customer)')
    parser.add_argument('-o', '--output', type=str, default='test', help='output dir name')
    return parser.parse_args()

args = parse_arguments()
refiner = Refiner(args)
refiner.run()