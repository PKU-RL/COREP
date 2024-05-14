# 📄 Tackling Non-Stationarity in Reinforcement Learning via Causal-Origin Representation
      
The official repository of our [[Paper]](https://arxiv.org/pdf/2306.02747) at **ICML 2024**.

![COREP](imgs/COREP_framework.png)

COREP primarily employs a dual-GAT structure with a guided updating mechanism to learn a stable graph representation for states, termed as causal origin representation. By leveraging this representation, the learned policy exhibits resilience to non-stationarity. The overall framework of COREP is illustrated in the above figure.

## Installation

The main requirements can be found in `requirements.txt`. We also use the `dmc2gym` library to convert the deepmind control environment into an openai gym interface.

To install the requirements, you can follow the instructions below:

```
conda create -n COREP python=3.10 -y
conda activate COREP
pip install -r requirements.txt
pip install -e dmc2gym
```

## Running experiments

To evaluate COREP on the deepmind control environment, run

```
python main.py --env-type <env_name>
``` 

which will use hyperparameters from `config/args_<env_name>.py`. 

To reproduce the results in the paper, run the following commands:

```
python main.py --env-type cartpole_swingup
python main.py --env-type reacher_easy
python main.py --env-type reacher_hard
python main.py --env-type cup_catch
python main.py --env-type cheetah_run
python main.py --env-type hopper_stand
python main.py --env-type swimmer_swimmer6
python main.py --env-type swimmer_swimmer15
python main.py --env-type finger_spin
python main.py --env-type walker_walk
python main.py --env-type fish_upright
python main.py --env-type quadruped_walk
```

## Results

The results will be saved at `./logs`, to view the results on tensorboard run

```
tensorboard --logdir ./logs
```

## Acknowledgements

Our codebase is built upon the [VariBAD](https://github.com/lmzintgraf/varibad) repository, which we have modified to implement COREP.

## Citation

If you find our work useful in your research and would like to cite our project, please use the following citation:
```
@article{zhang2023tackling,
  title={Tackling Non-Stationarity in Reinforcement Learning via Causal-Origin Representation},
  author={Zhang, Wanpeng and Li, Yilin and Yang, Boyu and Lu, Zongqing},
  journal={arXiv preprint arXiv:2306.02747},
  year={2023}
}
```