# %% 
import time
from utils import *
from Config import Config
from work import work


if Config.sf>=11: logger.warning(f"WARNING: SF={Config.sf}, LDRO might be enabled.")

if __name__ == "__main__":
    script_path = __file__
    mod_time = os.path.getmtime(script_path)
    readable_time = time.ctime(mod_time)
    logger.warning(f"Last modified time of the script: {readable_time}")
    work(0, 0, "data/S0914_SF10_9041_at_ta_001.fc32")


# %%
