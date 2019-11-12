# Module to hold global variables for curvature computation functions.
# This is needed sincne functionality may be split over several modules

from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

event_writer: Optional[SummaryWriter] = None
project_name: Optional[str] = 'train_ciresan'  # project name to use for wandb logging
logdir_base: str = '/ncluster/runs'
run_name: Optional[str] = None  # run name to use, corresponds to logging dir and wandb run name
logdir: Optional[str] = None  # logdir
token_count: int = 0   # TODO(y): rename to global-step. Meaning is context-specific, in case of sequences it's number of tokens

args = None   #  global arg values
debug_dump_stats: bool = False   # print activations/backprops to console
debug_linalg_crashes: bool = False   # save matrices that cause linalg routines to crash


# debug_hard_crashes_on_nans: bool = True  # crash if encountering NaN

hacks_disable_hess = False


if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU")
else:
    device = torch.device('cpu')


def reset_global_step():
    global token_count
    token_count = 0


def increment_global_step(incr: int):
    global token_count
    token_count += incr


def get_global_step() -> int:
    return token_count


