# coding: utf-8
# 2020/5/8 @ tongshiwei

from longling import config_logging
from EduSim.Envs.KSS import KSSEnv, kss_train_eval, KSSAgent


env = KSSEnv()

kss_train_eval(
    KSSAgent(env.action_num),
    env,
    20, 4000,
    logger=config_logging(logger="kss", console_log_level="debug", level="debug"),
    level="summary",
    board_dir="./kss_logs"
)
