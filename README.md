Interpolation can hurt robust generalization even when there is no noise
================================================================
This repository contains the official code of

  > [Interpolation can hurt robust generalization even when there is no noise](https://arxiv.org/abs/2108.02883)
  (to appear in "Advances in Neural Information Processing Systems (NeurIPS), 2021")

by Konstantin Donhauser, Alexandru Țifrea, Michael Aerni, Reinhard Heckel, and Fanny Yang,
as well as the two related workshop papers
_"Surprising benefits of ridge regularization for noiseless regression"_
([pdf](pdf/surprising_benefits_of_ridge_regularization_for_noiseless_regression.pdf))
from the ICML 2021 [Overparameterization: Pitfalls & Opportunities](https://sites.google.com/view/icml2021oppo) workshop
and _["Maximizing the robust margin provably overfits on noiseless data"](https://openreview.net/forum?id=ujQKWaxFkrL)_
([pdf](pdf/maximizing_the_robust_margin_provably_overfits_on_noiseless_data.pdf))
from the ICML 2021 [A Blessing in Disguise: The Prospects and Perils of Adversarial Machine Learning](https://advml-workshop.github.io/icml2021/) workshop.

See the [Citations](#citations) section for details on how to cite our work.



Requirements
------------
### Initial setup
We manage dependencies using Conda.
The most lean way to use Conda is via
the [Miniconda](https://docs.conda.io/en/latest/miniconda.html) distribution.
After installing Conda,
the environment can be created by running

    conda env create -f environment.yml interpolation-robustness

from the command line.

Jupyter lab requires some additional
extensions which cannot be managed by Conda.
Make sure to first *activate the environment*
and then run

    jupyter labextension install jupyterlab-plotly@4.14.3 @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.3

from the command line.

Due to the way the CUDA dependency is implemented in jaxlib/XLA,
the following workaround is required to run JAX on a GPU.

1. Make sure the Conda environment is enabled.
2. Mimic the expected CUDA install directory structure by performing
   `mkdir -p "${CONDA_PREFIX}/cuda/nvvm/libdevice/" && ln -s "${CONDA_PREFIX}/lib/libdevice.10.bc" "${CONDA_PREFIX}/cuda/nvvm/libdevice/"`
   from the command line.
   This essentially links the libdevice library installed by CONDA
   to some mock directory structure expected by jaxlib/XLA.
3. Export the environment variable
   `export XLA_FLAGS=--xla_gpu_cuda_data_dir="${CONDA_PREFIX}/cuda/"`.
   This needs to be done before any JAX code is executed.
   The library is only searched once, hence, if you forget to export the
   environment variable and are using Jupyter,
   make sure to restart the notebook kernel after setting the environment variable.

Alternatively, you might be able to simply point `--xla_gpu_cuda_data_dir`
to your system's CUDA installation,
but there might be version mismatches that lead to crashes.

### Using the environment
The environment can be enabled and disabled
from the command line via

    # enable
    conda activate interpolation-robustness

    # disable
    conda deactivate

### Mosek DCP solver
[Mosek](https://www.mosek.com/) is a commercial convex programming solver
which can be used to optimize logistic regression problems
in the `logistic_regression_dcp` experiment.
We observed Mosek to be faster and more accurate than
the standard solvers supplied with CVXPY.

While Mosek is a commercial product,
they provide free academic licenses for research or educational purposes
at degree-granting academic institutions to faculty, students, and staff.
To obtain a license, visit https://www.mosek.com/products/academic-licenses/
and register using your email address from your academic institution.

The solver will by default assume the license file to be in `$HOME/mosek/mosek.lic`.
The path can be changed by setting the `MOSEKLM_LICENSE_FILE` environment variable.

A license is only required to run the `logistic_regression_dcp` experiment
with the Mosek solver.
All other code, including the other solvers in `logistic_regression_dcp`,
do not require a Mosek license file to be present anywhere on your computer.
If one of the experiment shell scripts contain `MOSEK` in their solver list,
the experiments can also be run if `MOSEK` is removed or replaced.
However, it might be that the results are less accurate or the
optimization problems cannot be solved by the other solvers.



Experiments
-----------
We provide one or more bash scripts to run each of our experiments.
We use [MLFlow](https://mlflow.org/) to collect experiment results
and Jupyter notebooks for evaluation.


### Training
Before running any experiments, make sure to install all dependencies
as described in the "Requirements" section.

All experiments are executable Python modules found under
`src/interpolation_robustness/experiments/`.
To run an experiment, make sure that `/src/`
is added to the `PYTHONPATH` environment variable
and run

      python -m interpolation_robustness.experiments.experiment_module

where `experiment_module` is the module corresponding to the experiment to execute.
All experiments provide a command line interface to vary _all_ parameters
that can be varied.

The other directories in this repository contain various artifacts related
to the different experiments, e.g. scripts to run experiments or Jupyter notebooks for evaluation.

We ran all gradient descent experiments on a single NVIDIA GeForce GTX 1080 Ti graphics card
and parallelized execution on an internal compute cluster.
However, each individual experiment runs within a few seconds to minutes,
even on an off-the-shelf CPU.


### Evaluation
Each experiment directory contains one or more Jupyter notebooks.
We use those notebooks to evaluate experiment results
and create all plots in our papers.

You can start Jupyter lab by issuing
    
    jupyter lab

on the command line.
If you are using MLFlow locally,
the experiment working directory must be the same as the Jupyter working
directory, else MLFlow fails to locate the experiment results.


### Environment variables
We simplify working with environment variables by using the `python-dotenv` package.
Place all environment variables in a file `.env` in the root of this repository.
You can use `template.env` as a template.



Datasets
--------
Additional to synthetic data, some experiments support the following image datasets:

- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [FashionMNIST](https://arxiv.org/abs/1708.07747) (MIT License)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)



Repository structure
--------------------
All datasets are by default stored in a subdirectory of `/data/`.
Logs and training artifacts are by default stored
in a subdirectory of `logs/`.
The directory `src/` contains all Python code,
with the subdirectory `src/interpolation_robustness`
being an importable Python module.
This structure allows us to reuse a lot of code between experiments.



Citations
---------
Please cite this code and our work as

    @misc{Donhauser21,
        title={Interpolation can hurt robust generalization even when there is no noise},
        author={Konstantin Donhauser and Alexandru Ţifrea and Michael Aerni and Reinhard Heckel and Fanny Yang},
        year={2021},
        eprint={2108.02883},
        archivePrefix={arXiv},
        primaryClass={stat.ML}
    }
