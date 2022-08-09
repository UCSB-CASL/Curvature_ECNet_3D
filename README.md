# Error-correcting neural networks for three-dimensional mean-curvature computation in the level-set method

**By [Luis Ángel Larios-Cárdenas](http://www.youngmin.com.mx) and [Frédéric Gibou](https://sites.engineering.ucsb.edu/~fgibou/index.html), 
_Computer Science and Mechanical Engineering Departments, [University of California, Santa Barabara](https://www.ucsb.edu)_**

**Tuesday, August 9, 2022**

***

This is the accompanying repository for our paper
["Machine learning algorithms for three-dimensional mean-curvature computation in the level-set method"](https://arxiv.org/abs/2201.12342).  
You may also check out the [preceding paper](https://arxiv.org/abs/2201.12342) and its [GitHub repository](https://github.com/UCSB-CASL/Curvature_ECNet_2D) 
for the two-dimensional case.  

### Contents

To prepare the environment, take a look at the [`requirements.txt`](requirements.txt) file.  The neural networks trained for each resolution 
are organized under the `models/η/#` folder, where $\eta$ represents the maximum level of refinement of the unit-cube octrees, and `#` is 
either `non-saddle` or `saddle`.  Furthermore, $\eta = 6, 7, ..., 11$, and $h = 1/(2^\eta)$ is the mesh size.  The shared files include:

1. `k_nnet.h5`: TensorFlow/Keras model in `HDF5` format (with detached optimizer).
2. `k_nnet.json`: Our custom `JSON` version of the neural network with hidden-layer weights encoded in `Base64` but decoded as `ASCII` text.
The `"output"` key refers to the last hidden layer.  It does not include the aditive neuron.
3. `k_pca_scaler.pkl`: PCA scaler stored in `pickle` format.
4. `k_pca_scaler.json`: `JSON` version of PCA scaler with plain-valued parameters.
5. `k_std_scaler.pkl`: Standard scaler in `pickle` format.
6. `k_std_scaler.json`: `JSON` version of standard scaler with plain-valued parameters.

### Testing data sets

We have included sample surface data under the [`data/`](data) folder for the grid resolution with $η = 6$.  Each interface has its own 
subdirectory with different experiment IDs.  You can test any of the trained neural networks on this data.  See the 
[`python/evaluating.py`](python/evaluating.py) script.  To try different experiments, see the multiline strings prefixed with `TODO:`.  You
will collect the results from executing [`python/evaluating.py`](python/evaluating.py) in the [`results/`](results) directory 

**Note**: these data sets include *6 samples per interface node*.  For non-saddle samples, we have already applied negative-mean-curvature
normalization but left their curvature data (i.e., `hk` ($h\kappa^*$), `ihk` ($h\kappa$), `h2kg` ($h^2\kappa_G^*$), and `ih2kg` ($h^2\kappa_G$)) 
with the sign intact so we can recover the right answer after computing $h\kappa_\mathcal{F}$.  When testing, you might get marginally 
different results than those reported in our paper.  This is because we took the results in the paper from the online inference computations 
performed in C++, where small floating-point variations are possible.

### Contact

Please reach out to [Luis Ángel](mailto:lal@cs.ucsb.edu) with further questions and/or concerns.  We can share the training data sets upon
reasonable request as they are quite large.  Thanks!
