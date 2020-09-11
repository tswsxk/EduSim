# coding: utf-8
# 2020/5/10 @ tongshiwei

from tensorboardX import SummaryWriter

__all__ = ["SummaryWriter", "board_episode_callback", "reward_summary_callback"]


def board_episode_callback(episode, reward, summary_writer: SummaryWriter):
    summary_writer.add_scalar(tag="episode_reward", scalar_value=reward, global_step=episode)


def reward_summary_callback(rewards, infos, logger):
    expected_reward = sum(rewards) / len(rewards)

    logger.info("Expected Reward: %s" % expected_reward)

    return expected_reward, infos
