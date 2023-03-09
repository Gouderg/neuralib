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

- [ ] Add test to all function in the code.
- [ ] Add a clipped function.
- [ ] Create a folder for all the main.


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

each neuron separately represents two classes — 0 for one of the classes, and a 1 for the other. A model with this type of output layer is called binary logistic regression. This single neuron could represent two classes like cat vs. dog, but it could also represent cat vs. not cat or any combination of 2 classes, and you could have many of these. For example, a model may have two binary output neurons. One of these neurons could be distinguishing between person/not person, and the other neuron could be deciding between indoors/outdoors. Binary logistic regression is a regressor type of algorithm, which will differ as we’ll use a sigmoid activation function for the output layer rather than softmax, and binary cross-entropy rather than categorical cross-entropy for calculating loss.