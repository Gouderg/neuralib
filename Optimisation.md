# Optimisation

## Round 1

Valgrind output before:
```
1.09861
==21060== 
==21060== HEAP SUMMARY:
==21060==     in use at exit: 0 bytes in 0 blocks
==21060==   total heap usage: 10,200,609 allocs, 10,200,609 frees, 2,477,856,368 bytes allocated
==21060== 
==21060== All heap blocks were freed -- no leaks are possible
==21060== 
==21060== For lists of detected and suppressed errors, rerun with: -s
==21060== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```
Execution time:
```
1.09861

real	0m3,301s
user	0m3,057s
sys	    0m0,240s

```

Valgrind output after changing for pointer:

```
1.09861
==19787== 
==19787== HEAP SUMMARY:
==19787==     in use at exit: 0 bytes in 0 blocks
==19787==   total heap usage: 9,000,599 allocs, 9,000,599 frees, 2,269,056,296 bytes allocated
==19787== 
==19787== All heap blocks were freed -- no leaks are possible
==19787== 
==19787== For lists of detected and suppressed errors, rerun with: -s
==19787== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```

Valgrind output after changing for reference:

```
==29004== HEAP SUMMARY:
==29004==     in use at exit: 0 bytes in 0 blocks
==29004==   total heap usage: 7,800,595 allocs, 7,800,595 frees, 1,918,656,296 bytes allocated
==29004== 
==29004== All heap blocks were freed -- no leaks are possible
```

Execution time :
```
real    0m3,203s
user    0m3,033s
sys     0m0,164s
```
## Round 2

Tester le code après avoir implémenté la régularization.  

Paramètres initiaux

```cpp
const int NB_EPOCH = 1000;
const int NB_POINT = 500;
const int NB_NEURON = 64;
```

```
real    0m41,571s
user    0m40,803s
sys     0m0,660s
```

En ajoutant la compilation 03.

```
real    0m40,877s
user    0m40,079s
sys     0m0,770s
```