# Efficient Bayesian Optimization for Target Vector Estimation

This repository accompanies the following paper:

Uhrenholt, A. and Jensen, B. S. "Efficient Bayesian Optimization for Target Vector Estimation". AISTATS 2019-22nd International Conference on Artificial Intelligence and Statistics. 2019.

## Overview

The Python module in this repository is an extension to the [Emukit](https://github.com/amzn/emukit) Bayesian optimization library. It is aimed at the task of estimating a target vector through a stochastic and blackboxed multi-output function, i.e. finding the inputs to the function which produces a pre-specified target vector.

![](https://raw.githubusercontent.com/akuhren/target_vector_estimation/master/imgs/opt_example.gif)

### Installation

To install the module and all dependencies, use pip as follows:

`pip install git+https://github.com/akuhren/target_vector_estimation`

### Usage

The notebook `Demonstration.ipynb` goes through the basic functionality of the code and illustrates graphically the difference between standard Bayesian optimization and the proposed approach.

The use of the code can be summarized as follows:
```
from l2_bayes_opt.acquisitions import \
    L2NegativeLowerConfidenceBound as L2_LCB

# Define objective function and initial sample points
f = ...
X0 = ...
Y0 = f(X0)

# Define Emukit input parameter space
parameter_space = ParameterSpace(...)

# Define GPy model and wrap for use in Emukit
model = GPRegression(X0, Y0)
model_wrapped = GPyModelWrapper(model)

# Set a target vector and instantiate the acquisition function
target = ...
acq = L2_LCB(model=model_wrapped, target=target)

# Create and run the Bayesian optimization loop for 10 iterations
bayesopt_loop = BayesianOptimizationLoop(
    model=model_wrapped, space=parameter_space, acquisition=acq)
bayesopt_loop.run_loop(f, 10)
```

## Background

Bayesian optimization is an established and theoretically well-founded approach to optimizing functions that are non-differentiable, stochastic, and expensive to evaluate. By fitting a probabilistic model to noisy observations of the unknown function, the predictive distribution over unseen outputs can be used for optimally selecting where to sample next.

A slight variation of this problem is that of approximating a target vector by querying a multi-output function. Say we wish to manufacture an implant for a patient and that this implant, once inserted, should produce certain blood flow measurements [Perdikaris et al., 2016]. Here the function inputs are the implant design, the function outputs are the observed blood flow measurements, and the target vector is optimal the blood flow measurements, as defined by medical experts.

One approach in approximating the target vector through Bayesian optimization is to minimize the sum of squares between observed output and target using a standard surrogate model such as a Gaussian process. This is the approach taken in [Perdikaris et al., 2016]. However, this implies two major caveats: 1) The individual outputs of the unknown function are never observed directly, and 2) we model the distance between two vectors with a Gaussian predictive distribution. The latter point is critical since the distribution over a distance is non-negative and asymmetrical, both of which qualities are not elicited by the normal predictive.

In the proposed approach we instead model each output dimension of the unknown function with a Gaussian process and infer a noncentral chi-squared distribution over the distance to the target. The optimization rely on established acquisition functions (either Expected Improvement or Lower Confidence Bound) that have been revised for the new predictive distribution. This allows for a more precise noise model and a more informed method for choosing new sampling points, yielding a demonstrably better optimization procedure.

## References

P. Perdikaris and G. E. Karniadakis.  Model inversion via multi-fidelity Bayesian optimization: A new paradigm for parameter estimation in haemodynamics, and beyond. Journal of *The Royal Society Interface*, 13(118):20151107, 2016.
