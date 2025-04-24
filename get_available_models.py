import sys
import os
# Add the path to the project root directory to the system path
path = os.getcwd()
sys.path.append(path)
sys.path.append(path + '/src/pytorch-project-base/')

import src as src
from src import *
from models import *
from utils.registry import MODEL_REGISTRY


print(list(MODEL_REGISTRY))