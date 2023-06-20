# Kernelized Bayesian Quadrature
Code for TMLR paper [*On Average-Case Error Bounds for Kernel-Based Bayesian Quadrature*](https://openreview.net/pdf?id=JJrKbq35l4)

## Dependencies

-	Python 3
-	NumPy
-	SciPy 
-	Scikit-Learn
-	Matplotlib
-	Pandas
-	Jupyter Lab
-	Seaborn

## Run the experiments

All experiments can be run in `main.ipynb`.  To customize your own run, modify the arguments of the following function inside the Jupyter notebook:

```python
run(data_dir, func_name, alg_list, noise_level, split)
```

*Input* arguments for the above function:

- 	**`data_dir`**:  The directory to save the data
- 	**`func_name`**:  Specify the function name; see `BQ_code/utils.py` for details
- 	**`alg_list`**:  List of algorithm names, supports `mc`, `mvs-mat`, `mvs-mat-mc`, `mvs-se` and `mvs-se-mc`
-	 **`noise_level`**:  Specify the noise corruption level, e.g., `1e-2`
-	 **`split`**:  Any number between 0 and 1

## Visualization

To visualize the results, open and execute `plot.ipynb`.  The main function is:

```python
plot(func_name, noise_level, alg_list, legend_loc, y_range, std_scale)
```

*Input* arguments for the above function:

- 	**`func_name`**, **`noise_level`** and  **`alg_list`** are the same as the `run` function
- 	**`legend_loc`**:  Location of the legend, e.g., `upper right`
- 	**`y_range`**:  Specify the range of the Y-axis manually if demanded
- 	**`std_scale`**:  Scale of the standard deviation









