"""
4DVARNN-DinAE (new modifications by mbeaucha) 
"""

__author		= "Maxime Beauchamp"
__version__ 		= "0.0.1"
__last_modification__	= "2020-07-01"

##################################
# Standard lib
##################################
import sys
import os
import shutil
import time as timer
import copy
from os.path import join as join_paths
from datetime import date, datetime, timedelta
import itertools
import warnings
import traceback
import re
import functools
import configparser
import builtins
import time
from time import sleep
import multiprocessing
# import mkl
import cv2
from tqdm import tqdm
from collections import OrderedDict
import pickle
import argparse
import ruamel.yaml
from pathlib import Path

assert sys.version_info >= (3,5), "Need Python>=3.6"

##################################
# Config
##################################
dirs = {}

# Define paths
datapath="/users/local/DATA/OSSE/"
basepath="/users/local/q20febvr/4dnvarnn/"

print("Initializing 4DVARNN-DinAE libraries...",flush=True)

##################################
# Scientific and mapping
##################################
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import pandas as pd
import shapely
from shapely import wkt
#import geopandas as gpd
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.sparse import diags
from scipy.stats import multivariate_normal
from scipy.ndimage.morphology import distance_transform_edt as bwdist
import scipy.ndimage as nd
from scipy.interpolate import RegularGridInterpolator
import skill_metrics as sm
import xarray as xr
from netCDF4 import Dataset
from pyflann import *

##################################
# Tools
##################################
import torch
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
import torch.nn.functional as F
sys.path.insert(0,f'{basepath}/dinae_4dvarnn/mods')
sys.path.insert(0,f'{basepath}/dinae_4dvarnn/mods/utils')
sys.path.insert(0,f'{basepath}/dinae_4dvarnn/mods/utils/utils_nn')
sys.path.insert(0,f'{basepath}/dinae_4dvarnn/mods/utils/utils_solver')
from .mods.import_Datasets_OSSE  import *
from .mods.import_Datasets_OSE   import *
from .mods.define_Models         import *
from .mods.define_Classifiers    import *
from .mods.learning_OSSE         import *
from .mods.learning_OSE         import *
from .mods.utils.tools           import *
from .mods.utils.yml_tools       import *
from .mods.utils.graphics        import *
print("...Done") # ... initializing Libraries



