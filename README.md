# READ ME

## Code used for the paper Learning from humans: combining imitation and deep reinforcement learning to accomplish human-level performance on a virtual foraging task 

### Please cite the paper if you used this code or any of its components

## Dependencies 

- pytorch
- scikit-learn

Note that this implementation does not support `cuda`. 

## Summary of the code

- `algorithms` contains our Pytorch implementations of the algortihms mentioned in the paper: TRPO, PPO, UATRPO, SAC, TD3 and Generative adversarial IL (GAIL)
- `BatchBW_HIL_torch.py` contains our implementation of MLE Imitation learning suitable also for Hierarchical Imitation Learning (https://arxiv.org/pdf/2103.12197.pdf)
- `models.py` contains the NN models used as parameterization for the policies
- `World.py` contains the environment
- `Plot.py` is used to draw the plots of the paper 

## Data Set

The human trajectories (Fig.9 in the Supplementary Material) are stored in the folder `Expert_Data`.

## Imitation Learning (Fig. 10)

```bash
python main.py --mode "HIL_ablation_study" --seed $1
```

## Reinforcement Learning (Fig. 4)

```bash
python main.py --number_options 1 --policy PPO --seed $1 --HIL --load_HIL_model 
```

## PPO ablation study (Fig. 11)

```bash
python main.py --mode HRL_ablation_study --policy PPO --seed $1 --HIL --load_HIL_model --load_HIL_model_expert_traj $2
```

## Imitation Learning Adversarial Reward (Fig. 12)

```bash
python main.py --number_options 1 --policy PPO --seed $1 --HIL --load_HIL_model --load_HIL_model_expert_traj $2 --adv_reward
```

## Imitation Learning + PPO Adversarial Reward (Fig. 13)

```bash
python main.py --number_options 1 --policy PPO --seed $1 --load_HIL_model_expert_traj $2 --adv_reward
```

## Imitation Learning Allocentric only (Fig. 14)

```bash
python main.py --mode "HIL_ablation_study_allocentric_only" --seed $1
```

## Imitation Learning + PPO Allocentric only (Fig. 15)

```bash
python main.py --mode HRL_ablation_study_allocentric_only --policy PPO --seed $1 --HIL --load_HIL_model --load_HIL_model_expert_traj $2
```
