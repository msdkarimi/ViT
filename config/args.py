import argparse
import os
import logging
import torch
from pathlib import Path

if not os.path.exists(os.path.join(os.getcwd(), "output")):
    os.mkdir(os.path.join(os.getcwd(), "output"))
    os.mkdir(os.path.join(os.path.join(os.getcwd(), "output"), "logs"))
    os.mkdir(os.path.join(os.path.join(os.getcwd(), "output"), "weights"))

parser = argparse.ArgumentParser()
parser.add_argument("--phase", help="whether train/eval/test", choices=["train", "validation", "test", "plot"])
parser.add_argument('--config', help="location of config file", default=os.path.join(os.getcwd(), 'config', 'config.yml'))
parser.add_argument('--stepSize', type=int, default=20, help='scheduler step size to reduce learning rate')
parser.add_argument('--cpu', action='store_true', help='If set, the experiment will run on the CPU.')
parser.add_argument("--output", help="location of outputDir, logs or weights", default=os.path.join(os.getcwd(), 'output'))
parser.add_argument("--checkpoint_file", help="location of outputDir, logs or weights", default=os.path.join(os.getcwd(), 'output', 'weights', 'best_checkpoint.pth'))

args = parser.parse_args()

if args.output != os.path.join(os.getcwd(), "output"):
    if not os.path.exists(os.path.join(os.getcwd(), args.output)):
        os.mkdir(os.path.join(os.getcwd(), args.output))
        if not os.path.exists(os.path.join(os.path.join(os.getcwd(), args.output), "logs")):
            os.mkdir(os.path.join(os.path.join(os.getcwd(), args.output), "logs"))
        if not os.path.exists(os.path.join(os.path.join(os.getcwd(), args.output), "weights")):
            os.mkdir(os.path.join(os.path.join(os.getcwd(), args.output), "weights"))
        args.output = os.path.join(os.getcwd(), args.output)

    else:
        args.output = os.path.join(os.getcwd(), args.output)


#
# if not args.cpu:
#     assert torch.cuda.is_available(), 'You need a CUDA capable device in order to run this experiment. See `--cpu` flag.'

args.cpu = True

logging.basicConfig(filename=f'{args.output}/logs/logs.txt',filemode="a", level=logging.INFO, format='> %(name)s | %(asctime)s | %(message)s')
logger = logging.getLogger(args.phase)

# logger.info("------------------------------START-------------------------")
#
# logger.info("-------------------------------END--------------------------")



