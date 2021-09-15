from functions import Trainer
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('-tb', '--train_batch', type=int, default=250, help='Train batch per epoch')
    parser.add_argument('-e', '--epoch', type=int, default=150, help='Train epoch')
    parser.add_argument('-s', '--shuffle', type=bool, default=True, help='Shuffle training data')
    parser.add_argument('-c', '--ckpt', type=str, default='test', help='Save ckpt name')
    parser.add_argument('-k', '--mesh_number', type=int, default=3, help='Number of output mesh')
    return parser.parse_args()

args = parse_arguments()
trainer = Trainer(args)
trainer.run()