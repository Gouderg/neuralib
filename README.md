# Neuralib

## TO DO

- [X] Add Tensor class to perform operation
- [X] Add Broadcasting addition

- [ ] Add Neural Network class
    - [X] Initialisation
    - [X] Forward Propagation
    - [ ] Back Propagation
    - [ ] Optimizer
    - [ ] Update
- [ ] Add Accuracy Graphics class
- [ ] Add Loss Graphics class
- [X] Create and plot dataset (gnuplot) 

## Equation use in the projet

### Layer dense - Forward pass

$$
    output = inputs \times weights + biases
$$

### Rectified Linear (ReLU) Activation Function - Forward pass

$$
    y = \left\{
            \begin{array}{ll}
                x & x \gt 0 \\
                0 & x \le 0
            \end{array}
        \right.
$$

### Softmax Activation Function - Forward pass

$$\Large
    S_{i,j} = \frac{e^{z_{i,j}}}{\sum_{l=1}^L e^{z_{i,j}}}
$$

### Categorical Cross-Entropy Loss

$$\Large
    L_i = -\sum_j y_{i,j} log(\hat{y}_{i,j})
$$

## Setup the project

Sous ubuntu :
```bash
sudo apt-get install gnuplot
```

In the project :
```bash
mkdir build
cd build
cmake ..
make
```