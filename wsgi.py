import os
import sys

project_home = os.path.dirname(__file__)

if project_home not in sys.path:
    sys.path.append(project_home)

os.environ["MLFLOW_TRACKING_URI"] = f"file:{project_home}/mlruns"
os.environ["MLFLOW_RUN_ID"] = "be260e8b8f3e4ae7b3018afa00a828ec"
os.environ["THRESHOLD"] = "0.5"

from src.api.app import app as application