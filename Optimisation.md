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
## Round 2 - Optimiser la classe Tensor.

Dans un premier temps, j'ai chercher l'implémentation de la matrice. Avan c'était un vector de vector. Maintenant, c'est un vecteur inline dont on calcule la position avec i et j

De plus, le produit vectoriel de l'implémentation inline ajoute un petit tricks qui consiste à intervertir le for k et le for j. Cela permet d'avoir moins de lecture mémoire inutile.



### Matrice de taille 512 par 512: 

```
Time difference = 1[s] vs 0[s]
Time difference = 1055[ms] vs 838[ms]
```

### Quand on ajoute l'optimizer -O sur une matrice de 2048 par 2048:

```
Time difference = 22[s] vs 9[s]
Time difference = 22891[ms] vs 9695[ms]
```

### Avec l'optimizer -O3:
```
Time difference = 23[s] vs 4[s]
Time difference = 23071[ms] vs 4473[ms]
```

Pour débugguer la mémoire de son code on peut utiliser la commande `valgrind --tool=cachegrind ./opti` puis `cg_annotate cachegrind.out.33577`

### Utilisation de la parallélisation

On peut ajouter la ligne suivante : 
```cpp
#include <omp.h>

#pragma omp parallel for private(i,j,k) shared(t1, t2, t3)
```

et dans le Makefile : `-fopenmp`

### Optimiser les flags de compilation

`gcc -c -Q -march=native --help=target`


