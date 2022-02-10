# MLP + Belief Propagation for Interpretable Node Classification

The graph belief propagation networks (GBPNs) is a family of graph neural networks models for node classification.
They are accurate, interpretable, and converge to a stationary solution as the number of BP steps increase.
On a high level, a GBPN first predicts an initial probabilities for each node's label using its features (with a MLP), then runs belief propagation to iteratively refine/correct the predictions.

![GBPN performance on PubMed](figs/demo.svg)


## Environment Setup

The implementations in this repository are tested under Python 3.8, PyTorch version 1.6.0, and Cuda 10.2.
To setup the environment, simply run the following:

```setup
bash install_requirements.sh
```

This command installs PyTorch Geometric, and compiles the sub-sampling algorithm written in C++.
Note PyTorch Geometric may fail to initialize ([issue](https://github.com/rusty1s/pytorch_geometric/issues/999)) if there are multiple versions of PyTorch installed.
Therefore, we highly recommend the users to start with a new conda environment.

## Basic Usage

A GBPN model consists of a MLP that maps features on each node to its self-potential, and a coupling matrix.
It can be defined in the same way as any PyTorch Module.

```python
model = GBPN(num_features, num_classes, dim_hidden=dim_hidden, num_layers=num_layers, 
                                        activation=nn.ReLU(), dropout_p=dropout_p, 
                                        lossfunc_BP=0, deg_scaling=False, learn_H=True)
```

In this example, _num\_features_ is the input dimension of the MLP, _num\_classes_ is the output dimension of the MLP, _dim\_hidden_ is the number of units per hidden layer in the MLP, and _num\_layers_ is the number of hidden layers in the MLP.
After defining the model, we can run GBPN inference as:

```python
log_b = model(x, edge_index, edge_weight=edge_weight, edge_rv=edge_rv, deg=deg, deg_ori=deg, K=5)
```
Here, _K_ controls the number of belief propagation steps.

## Reproducing Results in the Paper
To reproduce our main experimental results, one can simply run:
```bash
bash run.sh
```
which runs GBPN and baselines on all datasets.
To reproduce results for a particular dataset (e.g. sexual interaction):
```bash
make device="cuda" run_Sex
```
which gives (finishes running in 10 minutes):

| Model      | MLP    | GCN   | SAGE  | GAT   | GBPN-I | GBPN  |
| ---------- |------- | ----- | ----- | ----- | ------ | ----- |
| accuracy   | 74.5%  | 83.9% | 93.3% | 93.6% | 97.1%  | 97.4% |

## License
This project is release under the GNU GENERAL PUBLIC LICENSE.
