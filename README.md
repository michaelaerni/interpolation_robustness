Maximizing the robust margin provably overfits on noiseless data
================================================================
This repository contains the code related to the paper
["Maximizing the robust margin provably overfits on noiseless data"](https://openreview.net/forum?id=ujQKWaxFkrL)
by Konstantin Donhauser, Alexandru Tifrea, Michael Aerni, Reinhard Heckel, and Fanny Yang,
as well as code for some upcoming work.
To cite our work:

    @inproceedings{
    donhauser2021maximizing,
    title={Maximizing the robust margin provably overfits on noiseless data},
    author={Konstantin Donhauser and Alexandru Tifrea and Michael Aerni and Reinhard Heckel and Fanny Yang},
    booktitle={ICML 2021 Workshop on Adversarial Machine Learning},
    year={2021},
    url={https://openreview.net/forum?id=ujQKWaxFkrL}
    }


How to run experiments
----------------------
Before running any experiments, make sure to install all dependencies
as described in the "Dependency manangement" section.

All experiments are executable Python modules found under
`$REPO_ROOT/src/interpolation_robustness/experiments/`.
To run an experiment, make sure that `$REPO_ROOT/src/`
is added to the `PYTHONPATH` environment variable
and run

      python -m interpolation_robustness.experiments.experiment_module

where `experiment_module` is the module corresponding to the experiment to execute.
All experiments provide a command line interface to vary _all_ parameters
that can be varied.

The other directories in this repository contain various artifacts related
to the different experiments, e.g. scripts to run experiments or Jupyter notebooks for evaluation.


Dependency management
---------------------
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
the following workaround is required in order to have JAX run on a GPU.

1. Make sure the Conda environemnt is enabled.
2. Mimic the expeted CUDA install directory structure by performing
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
but there might be version mismatches and things might or might not work.

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
they provide free academic licences for research or educational purposes
at degree-granting academic institutions to faculty, students, and staff.
To obtain a licence, visit https://www.mosek.com/products/academic-licenses/
and register using your email address from your academic institution.

The solver will by default assume the licence file to be in `$HOME/mosek/mosek.lic`.
The path can be changed by setting the `MOSEKLM_LICENSE_FILE` environment variable.

A licence is only required to run the `logistic_regression_dcp` experiment
with the Mosek solver.
All other code, including the other solvers in `logistic_regression_dcp`,
do not require a Mosek licence file to be present anywhere on your computer.
If one of the experiment shell scripts contain `MOSEK` in their solver list,
the experiments can also be run if `MOSEK` is removed or replaced.
However, it might be that the results are less accurate or the
optimization problems cannot be solved by the other solvers.


Environment variables
---------------------
We simplify working with environment variables by using the `python-dotenv` package.
Place all environment variables in a file `.env` in the root of this repository.
You can use `template.env` as a template.


Experiments
-----------
All our experiments use one or more bash scripts to run experiments.
The results are collected using [MLFlow](https://mlflow.org/)
and plotted using Jupyter notebooks.
If you are using MLFlow locally,
the experiment working directory must be the same as the Jupyter working
directory, else MLFlow fails to locate the experiment results.

We ran all gradient descent experiments on a single NVIDIA GeForce GTX 1080 Ti graphics card
and parallelized execution on an internal compute cluster.
However, each individual experiment runs within a few seconds to minutes,
even on an off-the-shelf CPU.


Datasets
--------
Except synthetic data, some experiments support the following image datasets:

- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [FashionMNIST](https://arxiv.org/abs/1708.07747) (MIT License)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)


Repository and code structure
-----------------------------
Let `$REPO_ROOT` denote the directory containing this README.
All datasets are by default stored in a subdirectory of
`$REPO_ROOT/data/`.
Logs and training artifacts are by default stored
in a subdirectory of `$REPO_ROOT/logs/`.
The directory `$REPO_ROOT/src/` contains all code.
The subdirectory `$REPO_ROOT/src/interpolation_robustness` is an importable Python
module which keeps things modular and makes it very easy
to reuse components between experiments.
