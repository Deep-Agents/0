---
layout: post
title: Solving Pommerman with Deep Reinforcement Learning
---

Reinforcement learning has been used to solve a number of challenging games recently. That said, there are many games that are as of yet unsolved or require a lot of domain knowledge in order to create intelligent agents. Pommerman, a bomberman clone ([further described here](/pom_info/)) provides a simple environment with fun and intuitive dynamics which are surprisingly deep.

# Problem Summary

How can we train agents capable of working and communicating together in a 2v2 competition? That is one of the core questions of the Pommerman competition this year (2019). Each agent must learn to not only manage aggressive strategies that can result in accidental suicide, it must also learn to predict and account for the actions of its teammate and enemies.



# Our Approach

In general our approach can be summarized as: attempt to use existing agents and work from there. Failing that, we work on the problem from scratch.

## Existing Agent Use
The first place learning agent (4th overall) from last year, [Navocado](https://arxiv.org/abs/1812.07297), has a much stronger and more general approach to reinforcement learning agent development, using an ensemble of agents and hyperparameter search with some ideas that are similar to [IMAPLA](https://arxiv.org/abs/1802.01561) from Google's DeepMind, however it was neigh impossible to replicate their work with our time constraints as their code was not well documented and relies on a distributed computing setup with orchestrated Docker containers.

MAGNet was another reinforcement learning network for Pommerman we tried to use, but it also had serious compute requirements (it was an order of magnitude slower to train than other agents we tried) which we hoped to avoid with our own more lightweight approaches.

We spent a good portion of our time on [Skynet from Borealis AI](https://www.borealisai.com/en/blog/pommerman-team-competition-or-how-we-learned-stop-worrying-and-love-battle/) due to the quality of their code documentation. [Our codebase](https://github.com/Deep-Agents/Deep-Pommerman) was largely based around this agent which placed 2nd at NeurIPS 2018 for learning agents and 5th overall. However, as time went on, we found their strategy to be particularly unsatisfying, as it primarily relies on a [deterministic action filter](https://github.com/BorealisAI/pommerman-baseline) that prevents agents from making obviously poor choices most of the time (like bombing themselves). Unfortunately this results in agents that humans can consistently take advantage of after only a couple games. Teamwork is also only a loose concept, with Skynet team agents randomly thrashing/walking around if they get too close to each other. That said, we did train a few agents by combining their system with our improvements (see results below).


## Agents From Scratch

To build our own agents, we needed to investigate the most effective learning algorithms. We compared each learning algorithm on a simplified version of the environment that allowed us to compare algorithm performance quickly.
![RL algorithm comparison](../images/algo-comparison-legend.png)

PPO tended to do the best, so we settled on it for all of the agents we attempted to train.

## Challenges

We faced a number of challenges such as poor code documentation, parallelization issues, overfitting to strategies, and trouble learning complex strategies. What our agents learned was also something we had to watch closely, as they would often make serious strategic errors and have a hard time unlearning those behaviors.

### Strategy Overfitting

Agents trained on environments that were too consistent tended to come up with trivial strategies (move left, then up, then place bomb). We found it necessary to randomize agent starting positions to account for this issue.

### Complex Strategy Learning

In general, our agents did not seem to learn very complex strategies. It is unclear whether this was due to insufficient training time, poor reward engineering, or neural networks that were not deep enough. One avenue of future research could be to quantify how large a neural network should be in relation to the difficulty of the task.

### Bomb Handling Depression

Skynet agents all would get depressed and stop placing bombs after one their teammates died. We believe this was due to the way team spirit rewards were handled, but did not find a good way of getting rid of this behavior.
![](../images/skynet-depressed.gif)

### Exploiting bugs in enemies strategies
The built-in deterministic agent (SimpleAgent) has a few bugs in its AI that our agents learned to take advantage of in order to make it suicide (basically if you sit in a certain spot in relation to SimpleAgent, it will stay there and place a bomb on itself). 
![](../images/simpleAgent-bug.gif)
This is an example of our agent overfitting to the strategy of the enemy agent, as it is highly unlikely this precise strategy would work on any other agents. It was avoided by varying the enemy agents our agent had to train against (to a simple extent). This is a sentiment that [DeepMind's AlphaStar team](https://deepmind.com/blog/article/AlphaStar-Grandmaster-level-in-StarCraft-II-using-multi-agent-reinforcement-learning) also seems to agree with:
>The key insight of the League is that playing to win is insufficient: instead, we need both main agents whose goal is to win versus everyone, and also exploiter agents that focus on helping the main agent grow stronger by exposing its flaws, rather than maximising their own win rate against all players. Using this training method, the League learns all its complex StarCraft II strategy in an end-to-end, fully automated fashion



## Improvements

### Curricula

One of our primary successes was in the development of a curriculum of lessons that agents had to achieve a certain level of ability on in order to move to successive lessons. We did this by varying the environment clutter (destructible boxes, indestructible obstacles), agent starting position, and enemy agent difficulty/AI.

In our experiments with the board size curriculum, the 11x11 board was walled up with indestructible blocks, leaving only a smaller central portion exposed for the agents to move in. Agents were progressively trained from a minimum board size of 4x4 up to the maximum 11x11 board. 

![](../images/env-shaping.gif)

One advantage of implementing the board size curriculum in this manner was that the underlying observation space remained a constant 11x11 through all experiments, allowing the same agent to be trained and evaluated in different environment sizes. Below, we show the performance of the longest-trained agent in our experiments, at 300 million timesteps, on different board sizes. 

![](../images/WTLvsSize.png)

The agent trained extensively on a 4x4 board still showed respectable performance on larger board sizes, exhibiting a _graceful_ decline. 

![](../images/curric-box.gif)

### Network Architecture

Not all neural networks are equal! Some networks seemed to learn our curriculum with fewer episodes as compared to others. One key improvement we made was based on the positional relation of our observational data. Though our observation space is discrete, it can be thought of in a similar way as images, thus CNNs seem like a natural fit.
![](../images/CNN-arch.png)
Training efficiency increased notably with our CNN architecture as compared to our fully connected layer architecture.

# Results

We were able to beat the baseline deterministic agent, SimpleAgent, consistently with most of our trained agents. A detailed review of each agent's performance can be seen on our [Leaderboard](/leaderboards/). 

The Skynet agent we trained ourselves from scratch, combining their reward structure with our curriculum, also achieved marginally better performance than the agent they entered in last year's competition, though in matches against each other, both agents usually tie as the action filter they use tends to make them over-cautious. One interesting note is that our version tends to complete matches faster/is slightly more aggressive--this is likely because we had it train so much against non-moving agents with fewer timesteps given before the end of each episode.

# Lessons Learned

[Reinforcement learning is hard!](https://www.alexirpan.com/2018/02/14/rl-hard.html)

## Parallelize!

If we had realized the importance of distributed computing and parallelization sooner, there's a good chance we could have gotten much better results (though it likely would have cost a lot more $$$). It's highly recommended that if you're serious about reinforcement learning (and not taking months to train decent agents), you start with the mindset of distributed computing and parallelization from the very beginning. One recommended RL scaling framework is [Ray's RLlib](https://ray.readthedocs.io/en/latest/rllib.html).

## Do Imitation Learning First! (If Possible)

Only towards the end did we realize how much more effective we could have been if we had started with an agent that learned to imitate from some known decent agent (human or deterministic). Only towards the end of our time did we discover [this work](https://github.com/eugene/pommerman) which went through many of the challenges we did and settled on imitation learning for decent performance. We likely could have imitated the best agents in last year's competition to get our baseline agents, and then trained on multi-agent communication from there.


# References

1. Resnick, Cinjon, et al. "Pommerman: A multi-agent playground." arXiv preprint arXiv:1809.07124 (2018).
1. Osogami, Takayuki, and Toshihiro Takahashi. "Real-time tree search with pessimistic scenarios." arXiv preprint arXiv:1902.10870 (2019).
1. Gao, Chao, et al. "Skynet: A Top Deep RL Agent in the Inaugural Pommerman Team Competition." arXiv preprint arXiv:1905.01360 (2019).
1. Malysheva, Aleksandra, Daniel Kudenko, and Aleksei Shpilman. "MAGNet: Multi-agent Graph Network for Deep Multi-agent Reinforcement Learning."
1. Peng, Peng, et al. "Continual match based training in Pommerman: Technical report." arXiv preprint arXiv:1812.07297 (2018).
1. Shah, Dhruv, Nihal Singh, and Chinmay Talegaonkar. "Multi-Agent Strategies for Pommerman."
1. Kartal, Bilal, et al. "Safer Deep RL with Shallow MCTS: A Case Study in Pommerman." arXiv preprint arXiv:1904.05759 (2019).
1. Resnick, Cinjon, et al. "Backplay:" Man muss immer umkehren"." arXiv preprint arXiv:1807.06919 (2018).
1. Kapoor, Sanyam. "Multi-agent reinforcement learning: A report on challenges and approaches." arXiv preprint arXiv:1807.09427 (2018).
1. Zhou, Hongwei, et al. "A hybrid search agent in pommerman." Proceedings of the 13th International Conference on the Foundations of Digital Games. ACM, 2018.
1. Gao, Chao, et al. "On hard exploration for reinforcement learning: A case study in pommerman." Proceedings of the AAAI Conference on 1. Artificial Intelligence and Interactive Digital Entertainment. Vol. 15. No. 1. 2019.
1. Perez-Liebana, Diego, et al. "Analysis of Statistical Forward Planning Methods in Pommerman." Proceedings of the AAAI Conference on 1. Artificial Intelligence and Interactive Digital Entertainment. Vol. 15. No. 1. 2019.
1. Kartal, Bilal, Pablo Hernandez-Leal, and Matthew E. Taylor. "Terminal Prediction as an Auxiliary Task for Deep Reinforcement Learning." Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment. Vol. 15. No. 1. 2019.
