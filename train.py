import sys
sys.path.append('/DeepLearning/Users/kawai/Document/pytorch-project-base/')
sys.path.append('/DeepLearning/Users/kawai/Document/pytorch-project-base/src/pytorch-project-base/')
import os
import logging
import argparse
import json
import src as src
from src import *
from main import main


if __name__ == '__main__':
    # TODO: ログの出力先を変更する
    # TODO: ログレベルの設定ファイルからの変更
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Hello, world!")

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json', type=str, help='config file path (default: config.json)')
    args = parser.parse_args()

    config = json.load(open(args.config))
    main(config)