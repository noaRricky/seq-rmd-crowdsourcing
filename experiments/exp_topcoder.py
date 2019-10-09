from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch as T
import torch.optim as optim

from utils import get_log_dir
from datasets import SeqTopcoder
from models import FMLearner, TorchFM, TorchHrmFM, TorchPrmeFM, TorchTransFM
from models.fm_learner import simple_loss, simple_weight_loss, trans_loss, trans_weight_loss

# Define dataset setting
DEVICE = T.cuda.current_device()
BATCH = 2000
SHUFFLE = True
WORKERS = 0
NEG_SAMPLE = 5
REGS_PATH = Path("./inputs/topcoder/regs.csv")
CHAG_PATH = Path("./inputs/topcoder/challenge.csv")

# Read databunch
db = SeqTopcoder(regs_path=REGS_PATH, chag_path=CHAG_PATH)

# Define model setting
feat_dim = db.feat_dim
NUM_DIM = 124
INIT_MEAN = 0.1

# Define criterion
LINEAR_REG = 1
EMB_REG = 1
TRANS_REG = 1

callback_simple_loss = partial(simple_loss, LINEAR_REG, EMB_REG)
callback_trans_loss = partial(trans_loss, LINEAR_REG, EMB_REG, TRANS_REG)
callback_simple_weight_loss = partial(simple_weight_loss, LINEAR_REG, EMB_REG)
callback_trans_weight_loss = partial(trans_weight_loss, LINEAR_REG, EMB_REG,
                                     TRANS_REG)

# Learning settings
LEARNING_RATE = 0.001
DECAY_FREQ = 1000
DECAY_GAMME = 0.9
EPOCH = 5
"""
Train model
1、First train simple loss
2、Second train weight loss
"""
fm_model = TorchFM(feature_dim=feat_dim, num_dim=NUM_DIM, init_mean=INIT_MEAN)
adam_opt = optim.Adam(fm_model.parameters(), lr=LEARNING_RATE)
schedular = optim.lr_scheduler.StepLR(adam_opt,
                                      step_size=DECAY_FREQ,
                                      gamma=DECAY_GAMME)
fm_learner = FMLearner(fm_model, adam_opt, schedular, db)
fm_learner.compile(train_col='seq',
                   valid_col='seq',
                   test_col='seq',
                   loss_callback=callback_simple_loss)
fm_learner.fit(epoch=EPOCH, log_dir=get_log_dir('simple_topcoder', 'fm'))
del fm_model
T.cuda.empty_cache()

fm_model = TorchFM(feature_dim=feat_dim, num_dim=NUM_DIM, init_mean=INIT_MEAN)
adam_opt = optim.Adam(fm_model.parameters(), lr=LEARNING_RATE)
schedular = optim.lr_scheduler.StepLR(adam_opt,
                                      step_size=DECAY_FREQ,
                                      gamma=DECAY_GAMME)
fm_learner = FMLearner(fm_model, adam_opt, schedular, db)
fm_learner.compile(train_col='seq',
                   valid_col='seq',
                   test_col='seq',
                   loss_callback=callback_simple_weight_loss)
fm_learner.fit(epoch=EPOCH, log_dir=get_log_dir('weight_topcoder', 'fm'))
del fm_model
T.cuda.empty_cache()

# ====================================
# HRM model
# =======================================
hrm_model = TorchHrmFM(feature_dim=feat_dim,
                       num_dim=NUM_DIM,
                       init_mean=INIT_MEAN)
adam_opt = optim.Adam(hrm_model.parameters(), lr=LEARNING_RATE)
schedular = optim.lr_scheduler.StepLR(adam_opt,
                                      step_size=DECAY_FREQ,
                                      gamma=DECAY_GAMME)
fm_learner = FMLearner(hrm_model, adam_opt, schedular, db)
fm_learner.compile(train_col='seq',
                   valid_col='seq',
                   test_col='seq',
                   loss_callback=callback_simple_loss)
fm_learner.fit(epoch=EPOCH, log_dir=get_log_dir('simple_topcoder', 'hrm'))
del hrm_model
T.cuda.empty_cache()

hrm_model = TorchHrmFM(feature_dim=feat_dim,
                       num_dim=NUM_DIM,
                       init_mean=INIT_MEAN)
adam_opt = optim.Adam(hrm_model.parameters(), lr=LEARNING_RATE)
schedular = optim.lr_scheduler.StepLR(adam_opt,
                                      step_size=DECAY_FREQ,
                                      gamma=DECAY_GAMME)
fm_learner = FMLearner(hrm_model, adam_opt, schedular, db)
fm_learner.compile(train_col='seq',
                   valid_col='seq',
                   test_col='seq',
                   loss_callback=callback_simple_weight_loss)
fm_learner.fit(epoch=EPOCH, log_dir=get_log_dir('weight_topcoder', 'hrm'))
del hrm_model
T.cuda.empty_cache()

# ================================================================
# PRME Model
# ===============================================================
prme_model = TorchPrmeFM(feature_dim=feat_dim,
                         num_dim=NUM_DIM,
                         init_mean=INIT_MEAN)
adam_opt = optim.Adam(prme_model.parameters(), lr=LEARNING_RATE)
schedular = optim.lr_scheduler.StepLR(adam_opt,
                                      step_size=DECAY_FREQ,
                                      gamma=DECAY_GAMME)
fm_learner = FMLearner(prme_model, adam_opt, schedular, db)
fm_learner.compile(train_col='seq',
                   valid_col='seq',
                   test_col='seq',
                   loss_callback=callback_simple_loss)
fm_learner.fit(epoch=EPOCH, log_dir=get_log_dir('simple_topcoder', 'prme'))
del prme_model
T.cuda.empty_cache()

prme_model = TorchPrmeFM(feature_dim=feat_dim,
                         num_dim=NUM_DIM,
                         init_mean=INIT_MEAN)
adam_opt = optim.Adam(prme_model.parameters(), lr=LEARNING_RATE)
schedular = optim.lr_scheduler.StepLR(adam_opt,
                                      step_size=DECAY_FREQ,
                                      gamma=DECAY_GAMME)
fm_learner = FMLearner(prme_model, adam_opt, schedular, db)
fm_learner.compile(train_col='seq',
                   valid_col='seq',
                   test_col='seq',
                   loss_callback=callback_simple_weight_loss)
fm_learner.fit(epoch=EPOCH, log_dir=get_log_dir('weight_topcoder', 'prme'))
del prme_model
T.cuda.empty_cache()

# ================================================
# Trans Model
# ===============================================
trans_model = TorchTransFM(feature_dim=feat_dim,
                           num_dim=NUM_DIM,
                           init_mean=INIT_MEAN)
adam_opt = optim.Adam(trans_model.parameters(), lr=LEARNING_RATE)
schedular = optim.lr_scheduler.StepLR(adam_opt,
                                      step_size=DECAY_FREQ,
                                      gamma=DECAY_GAMME)
fm_learner = FMLearner(trans_model, adam_opt, schedular, db)
fm_learner.compile(train_col='seq',
                   valid_col='seq',
                   test_col='seq',
                   loss_callback=callback_trans_loss)
fm_learner.fit(epoch=EPOCH, log_dir=get_log_dir('simple_topcoder', 'trans'))
del trans_model
T.cuda.empty_cache()

trans_model = TorchTransFM(feature_dim=feat_dim,
                           num_dim=NUM_DIM,
                           init_mean=INIT_MEAN)
adam_opt = optim.Adam(trans_model.parameters(), lr=LEARNING_RATE)
schedular = optim.lr_scheduler.StepLR(adam_opt,
                                      step_size=DECAY_FREQ,
                                      gamma=DECAY_GAMME)
fm_learner = FMLearner(trans_model, adam_opt, schedular, db)
fm_learner.compile(train_col='seq',
                   valid_col='seq',
                   test_col='seq',
                   loss_callback=callback_trans_weight_loss)
fm_learner.fit(epoch=EPOCH, log_dir=get_log_dir('weight_topcoder', 'trans'))
del trans_model
T.cuda.empty_cache()
