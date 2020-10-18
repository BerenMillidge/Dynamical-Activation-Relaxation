# Dynamical-Activation-Relaxation
Repo for experiments for the paper: "Investigating the scalability and biological plausibility of the activation relaxation algorithm". 

## Installation and Usage
Simply `git clone` the repository to your home computer. The `main.py` file will run the main model to test whether the frozen feedfoward pass assumption can be relaxed. The cnn.py will run the CNN experiments.

## Requirements 

The code is written in [Python 3.x] and uses the following packages:
* [NumPY]
* [PyTorch] version 1.3.1
* [matplotlib] for plotting figures

## Citation

If you enjoyed the paper or found the code useful, please cite as: 

```
@article{millidge2020activation,
  title={Activation Relaxation: A Local Dynamical Approximation to Backpropagation in the Brain},
  author={Millidge, Beren and Tschantz, Alexander and Buckley, Christopher L and Seth, Anil},
  journal={arXiv preprint arXiv:2009.05359},
  year={2020}
}
```
