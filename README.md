Already run experiments:
1) default: ep:50k, eps: 0.025, alpha: 0.10: Q:-285.49, DP: -255.5
2) ep:50k, eps: 0.0125, alpha: 0.10 : Q: -301.72, DP: -255.41
3) ep:50k, eps: 0.00625, alpha: 0.10: Q: -276.53, DP:-255.41
4) ep:50k, eps: 0.00625, alpha: 0.10, eps_decay_steps: 2500000: Q: -289.84, DP: -255.41
5) ep: 50k, eps: 0.00625, alpha: 0.10, eps_decay_steps: 1250000: Q:-295, DP;-255.41
6) ep: 50k, eps: 0.00625, alpha: 0.10, eps_decay_steps: 625000: Q-284.57, DP:-255.41
7) ep: 50k, eps: 0.00625, alpha: 0.05, eps_decay_steps: 1000000: Q: -265.76, DP: -255.41
8) ep: 50k, eps: 0.00625, alpha: 0.10, eps_decay_steps: 1000000:  Q: -268.37, DP: -255.41
9) ep: 100k, eps: 0.00625, alpha: 0.05, eps_decay_steps: 1000000: Q: -258.11, DP: -255.41
10) ep: 200k, eps: 0.00625, alpha: 0.05, eps_decay_steps: 1000000: Q: -246.67, DP: -255.41
11) ep: 400k, eps: 0.00625, alpha: 0.10, eps_decay_steps: 1000000: YET

overall notes and trends: alpha means how much the policy explores. more alpha means more exploration.
More episodes mean more convergence. if q is better than Dp, then evaluation is too shallow. increase number of episodes in evaluation till DP is better than Q.
All these confirm that model is working correctly. Now, need to isolate the effect of each hyperparameter.

Task 1: Ask AI to generate a set of experiments that isolate the effect of each hyperparameter. See report for reference on which hyperparameter. Compare across defence cost and reward.
Task 2: Make Attacker in RL and rerun the above experiments to see the difference. (rn only defender is RL and attacker is strategic)
Task 3: Make interactive visualization. The user should be able to play somehow(maybe through input commands or something).
