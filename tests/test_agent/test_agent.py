# coding: utf-8
# create by tongshiwei on 2019/7/1


def test_agent(random_agent, learner):
    agent = random_agent
    agent.begin_episode(learner)
    agent.step_reward()
    learning_item = agent.step()
    agent.observe(learning_item, 0)
    learning_item = agent.step()
    assert isinstance(learning_item, int)
    agent.n_step(10)
    agent.episode_reward()
    agent.end_episode()
    agent.tune()
