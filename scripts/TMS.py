# coding: utf-8
# 2020/4/30 @ tongshiwei

from longling import config_logging
from EduSim.Envs.TMS import TMSAgent, TMSEnv, tms_train_eval


env = TMSEnv("binary", mode="me")

tms_train_eval(
    TMSAgent(env.action_num),
    env,
    2, 10000,
    logger=config_logging(logger="TMS", console_log_level="debug", level="debug"),
    level="summary",
    board_dir="./tms_logs"
)

# env = TMSEnv("tree")
#
# tms_train_eval(
#     TMSAgent(env.action_num),
#     env,
#     6, 10000,
#     logger=config_logging(console_log_level="debug", level="debug"), level="episode"
# )
