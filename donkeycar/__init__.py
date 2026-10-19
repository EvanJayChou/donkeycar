import os
import sys
from pyfiglet import Figlet
import logging

__version__ = '5.4.dev1'

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())

f = Figlet(font='speed')


print(f.renderText('Donkey Car'))
print(f'using donkey v{__version__} ...')

# DONKEY_ALLOW_UNSUPPORTED_PYTHON=1 is set by the Jetson inference image,
# whose JetPack 6 system Python is 3.10.
if (sys.version_info.major < 3 or sys.version_info.minor < 12) and \
        os.environ.get('DONKEY_ALLOW_UNSUPPORTED_PYTHON') != '1':
    msg = (f'Donkey requires Python 3.12+ (PC/Mac) and 3.13+ (Raspberry Pi). '
           f'You are using {sys.version}')
    raise ValueError(msg)

# The default recursion limits in CPython are too small.
sys.setrecursionlimit(10**5)

from .vehicle import Vehicle
from .memory import Memory
from . import utils
from . import config
from . import contrib
from .config import load_config
