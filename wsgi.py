import os
import sys

project_home = os.path.dirname(__file__)

if project_home not in sys.path:
    sys.path.append(project_home)

os.environ["MODEL_PATH"] = os.path.join(
    project_home,
    "mlruns",
    "249839535688655471",
    "be260e8b8f3e4ae7b3018afa00a828ec",
    "artifacts",
    "model",
    "model.pkl",
)
os.environ["THRESHOLD"] = "0.5"

from src.api.app import app as application