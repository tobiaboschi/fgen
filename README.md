# fgen


FILES DESCRIPTION:

    fgen_solver/gen_core.py:
      function to run the fgen algorithm for one fixed value of c_lam

    fgen_solver/fgen_path.py:
      function to run the fgen algorithm for a grid of c_lam and compute the tuning criteria for each of them.

    fgen_solver/auxiliary_functions.py
      contains the auxiliary functions called by fgen_core and fgen_path, including proximal operator functions and conjugate functions.

    expes/main_core.py:
      main file to run fgen_core and competitor solvers on synthetic data 

    expes/main_path.py:
      main file to run fgen_path and competitor solvers on synthetic data 




THE FOLLOWING PYTHON PACKAGES ARE REQUIRED:
  
    - numpy
    - Scikit-learn
    - scipy
    - tqdm
    - matplotlib
    - pandas
    - scikit-fda




TO RUN THE CODE: 

    1) open a python3.8 environment
    2) Install the package by running `pip install -e .` at the root of the repository, i.e. where the setup.py file is.
    3) Lunch the desired experiments, e.g. `python expes/main_core.py`




THE CODE FOLLOWS A NOTATION DIFFERENT FROM THE ONE OF THE PAPER. It follows the notation of the majority of optimization sofwtares:

    m: number of observations
    n: number of features 
    k: number of elements in each group
    A: desing matrix (m x n)
    b: response matrix (m x k)
    x: coefficient matrix (n x k) 
    y: dual variable 1 (m x k)
    z: dual variable 2 (n x k)

