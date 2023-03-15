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
- [X] Add Unit test for tensor class cxxtest
- [X] Add system to throw error and hanle assertion.
- [X] Binomial Unit test
- [X] Add a clipped function.
- [X] Create a folder for all the main.

- [ ] Add test to all function in the code.
    - [X] Activation ReLU
    - [X] Activation Softmax
    - [X] Activation Sigmoid
    - [X] Activation Linear
    - [X] Layer Dense
    - [X] Layer_Dropout
    - [X] Loss_CategoricalCrossentropy
    - [X] Loss_BinaryCrossentropy
    - [X] Loss_MeanSquaredError
    - [X] Loss_MeanAbsoluteError
    - [X] Activation_Softmax_Loss_CategoricalCrossentropy
    - [X] Optimizer_SGD
    - [X] Optimizer_Adagrad
    - [X] Optimizer_RMSProp
    - [X] Optimizer_Adam
- [X] Add reshape function
- [X] Add cout shape
- [ ] Refacto Loss (Ajouter des assert dans le forward et le backward)
- [ ] Améliorer le système d'affichage de graph
- [ ] Add struct parameters for layer_dense class



## Todo Clean code

- [ ] Do a makefile with specific flag who fit with AMD and Intel.
- [ ] set(CMAKE_CXX_FLAGS "-Wall -Wextra")



## Setup the project

Sous ubuntu :
```bash
sudo apt-get install gnuplot libboost-all-dev cxxtest
```

In the project :
```bash
mkdir build
cd build
cmake ..
make
```

To build all unit test, just write make at the source of the project



## Bibliographie

- https://valgrind.org/docs/manual/cg-manual.html
- https://vaibhaw-vipul.medium.com/matrix-multiplication-optimizing-the-code-from-6-hours-to-1-sec-70889d33dcfa
- https://wiki.gentoo.org/wiki/GCC_optimization/fr
- https://github.com/deftio/C-and-Cpp-Tests-with-CI-CD-Example
- https://stackoverflow.com/questions/65871948/same-random-numbers-in-c-as-computed-by-python3-numpy-random-rand
- https://cxxtest.com/
- http://cxxtest.com/guide.html
