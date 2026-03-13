import sys
import os

# chemin vers ton projet
project_home = os.path.dirname(__file__)

if project_home not in sys.path:
    sys.path.append(project_home)

from src.api.app import app as application