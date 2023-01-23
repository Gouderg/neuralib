# Neuralib

## TO DO

- [X] Add Tensor class to perform operation
- [X] Add Broadcasting addition

- [ ] Add Neural Network class
    - [X] Initialisation
    - [X] Forward Propagation
    - [X] Back Propagation
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

## Setup the project

Sous ubuntu :
```bash
sudo apt-get install gnuplot libboost-all-dev
```

In the project :
```bash
mkdir build
cd build
cmake ..
make
```