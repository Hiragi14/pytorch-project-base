import sys
import os
# Add the path to the project root directory to the system path
path = os.getcwd()
sys.path.append(path)
sys.path.append(path + '/src/pytorch-project-base/')
import os
import logging
import argparse
import json
import src as src
from src import *
from main import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json', type=str, help='config file path (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args = parser.parse_args()

    config = json.load(open(args.config))
    config.update({'resume': args.resume})
    main(config)