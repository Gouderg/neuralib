# Optimisation

## Round 1

Valgrind output before:
```
==21060== Memcheck, a memory error detector
==21060== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==21060== Using Valgrind-3.18.1 and LibVEX; rerun with -h for copyright info
==21060== Command: ./neuralib
==21060== 
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
==19787== Memcheck, a memory error detector
==19787== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==19787== Using Valgrind-3.18.1 and LibVEX; rerun with -h for copyright info
==19787== Command: ./neuralib
==19787== 
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


