Vectorized-CMRG
===============

Implements a vectorized CMRG based on [1].  Generates numbers from multiple streams simultaneously using SSE or AVX.  See [blog post](http://codehathi.com/2013/11/11/vectorized-random-number-generation/) for more details.

## Testing notes
I don't currently have access to a system with AVX support so I was not able to test the performance or correctness of the output for the AVX code.  I used a slightly different code using AVX that this code is based off a while back, but I haven't been able to test this one yet.

## To compile
### Compilers tested
Tested the SSE and non-vectorized of these on OS X 10.9, Ubuntu 10.10, Scientific Linux 5.7, and Cray Linux Environment.

* gcc 4.7-4.9
* Apple LLVM version 5.0 (clang-500.2.79) (based on LLVM 3.3svn)

The following compile lines show how to run the test code.  If you would like to use the RNG elsewhere, compile through normal means without the -DTESTING flag.

### SSE Support
```bash
gcc VectorizedCMRG.c -DTESTING -DCMRG_SSE -lm
```
### AVX Support
Note: not fully tested yet.
```bash
gcc VectorizedCMRG.c -DTESTING -DCMRG_AVX -lm
```
### No vectorization
```bash
gcc VectorizedCMRG.c -DTESTING -DCMRG_AVX -lm
```

## Performance
As noted above, I haven't tested the AVX version of this code, so below are only performance results for the non-vectorized and SSE versions.

![Runtime results](http://4.bp.blogspot.com/-Yra1TNfylk8/UoD3Ivcy5OI/AAAAAAAAAKE/5_uFfiKQI9w/s640/runtimes.png "Performance of SSE implementation")

## References
1. P. L'Ecuyer, "Good Parameters and Implementations for Combined Multiple Recursive Random Number Generators," Operations Research, vol. 47, no. 1, pp. 159-164, Feb. 1999.
