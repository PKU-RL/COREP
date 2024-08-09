# 📄 Tackling Non-Stationarity in Reinforcement Learning via Causal-Origin Representation
      
The official repository of our [[Paper]](https://openreview.net/pdf?id=WLGWMDtj8L) at **ICML 2024**.

![COREP](imgs/COREP_framework.png)

COREP primarily employs a dual-GAT structure with a guided updating mechanism to learn a stable graph representation for states, termed as causal origin representation. By leveraging this representation, the learned policy exhibits resilience to non-stationarity. The overall framework of COREP is illustrated in the above figure.

## Install

1. Clone this repository and navigate to LLaVA folder

```bash
git clone https://github.com/PKU-RL/COREP.git
cd COREP
```

2. Install Package

```bash
conda create -n COREP python=3.10 -y
conda activate COREP
pip install --upgrade pip
pip install -r requirements.txt
```

3. Install additional packages. We use the [dmc2gym](https://github.com/denisyarats/dmc2gym) library to convert the deepmind control environment into an openai gym interface.

```bash
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

Parts of the code are based on the [VariBAD](https://github.com/lmzintgraf/varibad) repository, which we have modified to implement COREP.

## Citation

If you find our work useful in your research and would like to cite our project, please use the following citation:
```
@InProceedings{zhang2024tackling,
  title={Tackling Non-Stationarity in Reinforcement Learning via Causal-Origin Representation},
  author={Zhang, Wanpeng and Li, Yilin and Yang, Boyu and Lu, Zongqing},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  pages={59264--59288},
  year={2024},
  volume={235},
  publisher={PMLR}
}
```
