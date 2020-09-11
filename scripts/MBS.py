# coding: utf-8
# 2020/5/13 @ tongshiwei


from longling import config_logging
from EduSim.Envs.MBS import EFCEnv, HLREnv, mbs_train_eval, MBSAgent

# env = EFCEnv(n_items=30)
env = HLREnv(n_items=30)


mbs_train_eval(
    MBSAgent(env.n_items),
    env,
    # 200, 100,
    10, 1,
    logger=config_logging(logger="mbs", console_log_level="debug", level="debug"),
    level="step",
    board_dir="./mbs_logs"
)
