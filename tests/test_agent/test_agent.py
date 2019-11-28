# coding: utf-8
# create by tongshiwei on 2019/7/1

from EduSim.Agent.base import return_key_agent_summary_episode


def test_agent(random_agent, learner):
    agent = random_agent
    with agent.episode(learner):
        agent.step_reward()
        learning_item = agent.step()
        agent.observe(learning_item, 0)
        learning_item = agent.step()
        assert isinstance(learning_item, int)
        agent.n_step(10)
        agent.episode_reward(10)
        assert isinstance(agent.path, list)
        assert set(return_key_agent_summary_episode) == set(agent.summary_episode())
        agent.tune()
