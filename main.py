# %% 
import time
from utils import *
from Config import Config
from work import work
from work_new import work_new


if Config.sf>=11: logger.warning(f"WARNING: SF={Config.sf}, LDRO might be enabled.")

if __name__ == "__main__":
    fstart = -40971.948630148894
    tstart =  4240090.873306715
    # work(fstart, 0, "data/test_1226")
    work_new(fstart, tstart, "data/test_1226")

# %%
