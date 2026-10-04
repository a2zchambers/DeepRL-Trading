# config.py
import os
import torch

# Dynamically calculate the path of the project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Resolve the path two directories above (../../)
PARENT_TWO_UP = os.path.dirname(os.path.dirname(BASE_DIR))

# Absolute paths to your database assets
#FUNDAMENTAL_DB_PATH = os.path.abspath(os.path.join(PARENT_TWO_UP, "trading_results.db"))
#PRICING_DB_PATH = os.path.abspath(os.path.join(PARENT_TWO_UP, "pricing_data.db"))
# Point precisely to the parent directory (one directory above)
#FUNDAMENTAL_DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "trading_results.db"))
#PRICING_DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "pricing_data.db"))

FUNDAMENTAL_DB_PATH = os.path.join(BASE_DIR, "trading_results.db")
PRICING_DB_PATH = os.path.join(BASE_DIR, "pricing_data.db")

# Universe Configuration
TICKERS = ["NVDA", "AMD", "INTC"]

# Reinforcement Learning Hyperparameters
LOOKBACK_WINDOW = 5
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
EPISODES = 20
NOISE_SCALE = 0.05

# MacBook Pro (M-Series) MPS Hardware Acceleration Activation
DEVICE = torch.device("mps" if torch.cuda.is_available() is False and torch.backends.mps.is_available() else "cpu")
