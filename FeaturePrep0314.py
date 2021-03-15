import json
from json.decoder import JSONDecodeError
import os
import glob
import pandas as pd
import numpy as np
from urllib.parse import urlparse


dir = os.getcwd()
in_files = glob.glob(dir+'/PrepData0313/'+'*.json')
out_dir = dir+'/PrepData0314/'
list_of_dct = []
unavailable_json = []

for file in filter_files:
    with open(file, 'r') as f:
        filename = os.path.basename(file)
        try:
            dct = json.load(f)
