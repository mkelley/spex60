# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import json

DEFAULTS = {
    'lincor max': 3500,
    'y range': [625, 1225],
    'x range': [1040, 1901],
    'step': 5,
    'bottom': 624,
    'top': 1223,
    'readnoise': 12,
    'gain': 1.5,
    'spextool_path': os.path.join(os.path.expanduser("~"), 'local', 'idl',
                                  'irtf', 'Spextool')
}


def find_config():
    """Locate the config file."""
    fn = os.path.join(os.path.expanduser("~"), '.config', 'spex60',
                      'spex60.config')
    if not os.path.exists(fn):
        create_config(fn)

    return fn


def create_config(fn):
    path = os.path.dirname(fn)
    d = path.split(os.path.sep)
    for i in range(len(d)):
        x = os.path.sep.join(d[:i+1])
        if len(x) == 0:
            continue
        if not os.path.exists(x):
            os.mkdir(x)

    if not os.path.exists(fn):
        with open(fn, 'w') as outf:
            json.dump(DEFAULTS, outf)


config_file = find_config()
if not os.path.exists(config_file):
    create_config(config_file)

with open(config_file, 'r') as inf:
    config = json.load(inf)
