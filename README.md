# Neuralib

## TO DO

- [X] Add Tensor class to perform operation
- [X] Add Broadcasting addition
- [X] Add Neural Network class
    - [X] Initialisation
    - [X] Forward Propagation
    - [X] Back Propagation
    - [X] Optimizer
    - [X] Update
- [X] Add Accuracy Graphics class
- [X] Add Loss Graphics class
- [X] Create and plot dataset (gnuplot) 
- [ ] set(CMAKE_CXX_FLAGS "-Wall -Wextra")
- [ ] Do a makefile with specific flag who fit with AMD and Intel.
- [ ] Add Unit test for tensor class 
- [ ] Rearrange to make tensor class file readeable.

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

## Bibliographie

- https://valgrind.org/docs/manual/cg-manual.html
- https://vaibhaw-vipul.medium.com/matrix-multiplication-optimizing-the-code-from-6-hours-to-1-sec-70889d33dcfa
- https://wiki.gentoo.org/wiki/GCC_optimization/fr
- https://github.com/deftio/C-and-Cpp-Tests-with-CI-CD-Example