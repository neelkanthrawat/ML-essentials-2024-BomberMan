import logging
from pathlib import Path

from fallbacks import pygame

# Game properties
# board size (a smaller board may be useful at the beginning)
COLS = 9
ROWS = 9
SCENARIOS = {
    # modes useful for agent development
	"empty": {
        "CRATE_DENSITY": 0, 
        "COIN_COUNT": 0 
    },
    "coin-heaven": {
        "CRATE_DENSITY": 0,
        "COIN_COUNT":15 #6#11[64]# 12 #trained on 7x7. with 150 game steps
    },
    "loot-crate": { 
        "CRATE_DENSITY":0.45,#0.35,#0.15,#0.08,# 0.75, 
        "COIN_COUNT": 15#14#20 # earlier it was 12
    }, 
    # this is the tournaments game mode
    "classic": {
        "CRATE_DENSITY": 0.45,#0.75,
        "COIN_COUNT": 9
    }
    # Feel free to add more game modes and properties
    # game is created in environment.py -> BombeRLeWorld -> build_arena()
}
MAX_AGENTS = 4

# Round properties
MAX_STEPS = 400### note: choose large number of steps for training. I chose 150 and then it worked well

# GUI properties
GRID_SIZE = 30
WIDTH = 1000
HEIGHT = 600
GRID_OFFSET = [(HEIGHT - ROWS * GRID_SIZE) // 2] * 2

ASSET_DIR = Path(__file__).parent / "assets"

AGENT_COLORS = ['blue', 'green', 'yellow', 'pink']

# Game rules
BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2  # = 1 of bomb explosion + N of lingering around

# Rules for agents
TIMEOUT = 0.5
TRAIN_TIMEOUT = float("inf")
REWARD_KILL = 5
REWARD_COIN = 1

# User input
INPUT_MAP = {
    pygame.K_UP: 'UP',
    pygame.K_DOWN: 'DOWN',
    pygame.K_LEFT: 'LEFT',
    pygame.K_RIGHT: 'RIGHT',
    pygame.K_RETURN: 'WAIT',
    pygame.K_SPACE: 'BOMB',
}

# Logging levels
LOG_GAME = logging.INFO
LOG_AGENT_WRAPPER = logging.INFO
LOG_AGENT_CODE = logging.DEBUG
LOG_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
