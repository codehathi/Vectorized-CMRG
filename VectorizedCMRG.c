/******************************************************************************/
//  COPYRIGHT 2013, Codehathi
//  Author: Codehathi (codehathi@gmail.com)
//  File: VectorizedCMRG.h
//  Date: 09 Nov 2013
//  Version: 1.0
//  Description: CMRG for multiple independent RNG streams using vector units
//    P. L'Ecuyer, "Good Parameters and Implementations for Combined Multiple
//    Recursive Random Number Generators," Operations Research, vol. 47, no. 1,
//    pp. 159-164, Feb. 1999.
//
/******************************************************************************/
#include <stdlib.h>
#include <math.h>
#include "Logging.h"

#if defined CMRG_SSE
#ifdef __SSE4_1__
#include <smmintrin.h>
#else
#include <emmintrin.h>
#endif // end sse 4.1 check

typedef __m128d _VEC_TYPE;
typedef __m128d _MASK_TYPE;
#define ALIGN_AMOUNT 16
#define VECTOR_WIDTH 128
#define vec_set1(x)                     _mm_set1_pd(x)
#define vec_mult(x,y)                   _mm_mul_pd(x,y)
#define vec_sub(x,y)                    _mm_sub_pd(x,y)
#define vec_add(x,y)                    _mm_add_pd(x,y)
#define vec_cast_int(x)                 _mm_cvttpd_epi32(x)
#define vec_cast_double(x)              _mm_cvtepi32_pd(x)
#define vec_less_than(x,y)              _mm_cmplt_pd(x,y)
#define vec_less_equal_than(x,y)        _mm_cmple_pd(x,y)
#define vec_and(x, y)                   _mm_and_pd(x,y)

#elif defined CMRG_AVX

#include <immintrin.h>
typedef __m256d _VEC_TYPE;
typedef __m256d _MASK_TYPE;
#define ALIGN_AMOUNT 32
#define VECTOR_WIDTH 256
#define vec_set1(x)                     _mm256_set1_pd(x)
#define vec_mult(x,y)                   _mm256_mul_pd(x,y)
#define vec_sub(x,y)                    _mm256_sub_pd(x,y)
#define vec_add(x,y)                    _mm256_add_pd(x,y)
#define vec_cast_int(x)                 _mm256_cvttpd_epi32(x)
#define vec_cast_double(x)              _mm256_cvtepi32_pd(x)
#define vec_less_than(x,y)              _mm256_cmp_pd(x, y, _CMP_LT_OQ)
#define vec_less_equal_than(x,y)        _mm256_cmp_pd(x, y, _CMP_LE_OQ)
#define vec_and(x, y)                   _mm256_and_pd(x,y)

#else 

#include <float.h>
#include <stdint.h>
typedef double _VEC_TYPE;
typedef double _MASK_TYPE;
#define ALIGN_AMOUNT 8
#define VECTOR_WIDTH 64
#define vec_set1(x)                     (x)
#define vec_mult(x,y)                   ((x)*(y))
#define vec_sub(x,y)                    ((x)-(y))
#define vec_add(x,y)                    ((x)+(y))
#define vec_cast_int(x)                 ((int)(x))
#define vec_cast_double(x)              ((double)(x))
#define vec_less_than(x,y)              ((x)<(y)?-DBL_MAX:0x0)
#define vec_less_equal_than(x,y)        ((x)<=(y)?-DBL_MAX:0x0)

double and_doubles(double x, double y) {
  uint64_t temp = (*(uint64_t*)&x) & (*(uint64_t*)&y);
  return *(double*)&temp;
}

#define vec_and(x, y)                   and_doubles(x,y)

#endif

#if defined CMRG_SSE || defined CMRG_AVX
#define state_idx(stream, num)                                          \
  ((stream)%(NUM_ELEMENTS) + 6*(NUM_ELEMENTS)*(int)floor((stream)/(NUM_ELEMENTS)) + (num)*(NUM_ELEMENTS))
#define vec_idx(stream_group) ((stream_group)*6)

#else

#define state_idx(stream, num) ((stream)*6 + (num))
#define vec_idx(stream_group)  ((stream_group)*(6))
#endif

#define __CMRG_ZM1 4294967087.0
#define __CMRG_ZM2 4294944443.0
#define __CMRG_RM1 (1.0 / __CMRG_ZM1)
#define __CMRG_RM2 (1.0 / __CMRG_ZM2)

#define NUM_ELEMENTS ((VECTOR_WIDTH)/(8*8))


// global variable for internal states
double* rstatv;
int vec_cmrg_num_streams = 0;
_VEC_TYPE  rscale_vec    __attribute__ ((aligned (ALIGN_AMOUNT)));


// uniform rng for vectors.  each element of the vector is a independent stream
_VEC_TYPE vectorized_cmrg_rand_group(_VEC_TYPE* rstate)
{
  _VEC_TYPE z                                   __attribute__ ((aligned (ALIGN_AMOUNT)));
  _VEC_TYPE xx1                                 __attribute__ ((aligned (ALIGN_AMOUNT)));
  _VEC_TYPE temp                                __attribute__ ((aligned (ALIGN_AMOUNT)));
  const _VEC_TYPE scalar_vec_multiplier1        __attribute__ ((aligned (ALIGN_AMOUNT))) = vec_set1(1403580.0);
  const _VEC_TYPE scalar_vec_multiplier2        __attribute__ ((aligned (ALIGN_AMOUNT))) = vec_set1(810728.0);
  const _VEC_TYPE scalar_vec_multiplier3        __attribute__ ((aligned (ALIGN_AMOUNT))) = vec_set1(527612.0);
  const _VEC_TYPE scalar_vec_multiplier4        __attribute__ ((aligned (ALIGN_AMOUNT))) = vec_set1(1370589.0);
  const _VEC_TYPE RM1                           __attribute__ ((aligned (ALIGN_AMOUNT))) = vec_set1(__CMRG_RM1);
  const _VEC_TYPE RM2                           __attribute__ ((aligned (ALIGN_AMOUNT))) = vec_set1(__CMRG_RM2);
  const _VEC_TYPE ZM1                           __attribute__ ((aligned (ALIGN_AMOUNT))) = vec_set1(__CMRG_ZM1);
  const _VEC_TYPE ZM2                           __attribute__ ((aligned (ALIGN_AMOUNT))) = vec_set1(__CMRG_ZM2);
  const _VEC_TYPE zero                          __attribute__ ((aligned (ALIGN_AMOUNT))) = vec_set1(0.0);
  const _VEC_TYPE correction1                   __attribute__ ((aligned (ALIGN_AMOUNT))) = vec_set1(4294967087.0);
  const _VEC_TYPE correction2                   __attribute__ ((aligned (ALIGN_AMOUNT))) = vec_set1(4294944443.0);

  // xx1 = 1403580.0 * rstate[1] - 810728.0 * rstate[0];
  xx1  = vec_mult(scalar_vec_multiplier1, rstate[1]);
  temp = vec_mult(scalar_vec_multiplier2, rstate[0]);
  xx1 = vec_sub(xx1, temp);

  // rstate[0] = rstate[1]; 
  rstate[0] = rstate[1];

  // rstate[1] = rstate[2]; 
  rstate[1] = rstate[2];

  // rstate[2] = (xx1 - (int)( xx1 * __CMRG_RM1 ) * __CMRG_ZM1); 
  temp = vec_cast_double(vec_cast_int(vec_mult(RM1, xx1)));
  temp = vec_mult(ZM1, temp);
  rstate[2] = vec_sub(xx1, temp);

  // if(rstate[2] < 0.0) rstate[2] += 4294967087.0; 
  _MASK_TYPE mask = vec_less_than(rstate[2], zero);
  temp = vec_and(mask, correction1);
  rstate[2] = vec_add(rstate[2], temp);

  // xx1 = 527612.0 * rstate[5] - 1370589.0 * rstate[3]; 
  xx1 = vec_mult(scalar_vec_multiplier3, rstate[5]);
  temp = vec_mult(scalar_vec_multiplier4, rstate[3]);
  xx1 = vec_sub(xx1, temp);

  // rstate[3] = rstate[4]; 
  rstate[3] = rstate[4];

  // rstate[4] = rstate[5]; 
  rstate[4] = rstate[5];

  // rstate[5] = (xx1 - (int)( xx1 * __CMRG_RM2 ) * __CMRG_ZM2); 
  temp = vec_cast_double(vec_cast_int(vec_mult(RM2, xx1)));
  temp = vec_mult(ZM2, temp);
  rstate[5] = vec_sub(xx1, temp);

  // if (rstate[5] < 0.0) rstate[5] += 4294944443.0; 
  mask = vec_less_than(rstate[5], zero);
  temp = vec_and(mask, correction2);
  rstate[5] = vec_add(rstate[5], temp);

  // xx1=rstate[2]-rstate[5]; 
  xx1 = vec_sub(rstate[2], rstate[5]);

  // z = (xx1 - (int)( xx1 * __CMRG_RM1) * __CMRG_ZM1); 
  temp =vec_cast_double(vec_cast_int( vec_mult(RM1, xx1)));
  temp = vec_mult(ZM1, temp);
  z = vec_sub(xx1, temp);

  // if (z <= 0.0) z += 4294967087.0; 
  mask = vec_less_equal_than(z, zero);
  temp = vec_and(mask, correction1);
  z = vec_add(z, temp);

  // z *= rscale; 
  return vec_mult(z, rscale_vec);
}

// computes a*s % m without overflow using alg in extended version of
// P. L'Ecuyer  et al., Operations Research, volume 50, p. 1073 (2002).
double __vectorized_cmrg_init_mulmod(double a,
                                     double s,
                                     double zm)
{
  double u;
  double abig;
  double asmall;
  double two17 = 131072.0;
  asmall = fmod(a, two17);
  abig = (a-asmall)/two17;

  u = abig * s;

  u = fmod(u,zm) * two17 + asmall * s;
  return fmod(u, zm);
}

// Skips over seeds to get non-overlapping streams 
void __vectorized_cmrg_init_rskip(double* rstate)
{
  double xx1;
  double xx2;
  double xx3;
  double yy1;
  double yy2;
  double yy3;

  xx1 = __vectorized_cmrg_init_mulmod(rstate[0], 2427906178.0, 4294967087.0);
  xx2 = __vectorized_cmrg_init_mulmod(rstate[1], 3580155704.0, 4294967087.0);
  xx3 = __vectorized_cmrg_init_mulmod(rstate[2],  949770784.0, 4294967087.0);

  yy1 = fmod(xx1 + xx2 + xx3, 4294967087.0);

  xx1 = __vectorized_cmrg_init_mulmod(rstate[0],  226153695.0, 4294967087.0);
  xx2 = __vectorized_cmrg_init_mulmod(rstate[1], 1230515664.0, 4294967087.0);
  xx3 = __vectorized_cmrg_init_mulmod(rstate[2], 3580155704.0, 4294967087.0);

  yy2 = fmod(xx1 + xx2 + xx3, 4294967087.0);

  xx1 = __vectorized_cmrg_init_mulmod(rstate[0], 1988835001.0, 4294967087.0);
  xx2 = __vectorized_cmrg_init_mulmod(rstate[1],  986791581.0, 4294967087.0);
  xx3 = __vectorized_cmrg_init_mulmod(rstate[2], 1230515664.0, 4294967087.0);

  yy3 = fmod(xx1 + xx2 + xx3, 4294967087.0);

  rstate[0] = yy1;
  rstate[1] = yy2;
  rstate[2] = yy3;

  xx1 = __vectorized_cmrg_init_mulmod(rstate[3], 1464411153.0, 4294944443.0);
  xx2 = __vectorized_cmrg_init_mulmod(rstate[4],  277697599.0, 4294944443.0);
  xx3 = __vectorized_cmrg_init_mulmod(rstate[5], 1610723613.0, 4294944443.0);

  yy1 = fmod(xx1 + xx2 + xx3, 4294944443.0);

  xx1 = __vectorized_cmrg_init_mulmod(rstate[3],   32183930.0, 4294944443.0);
  xx2 = __vectorized_cmrg_init_mulmod(rstate[4], 1464411153.0, 4294944443.0);
  xx3 = __vectorized_cmrg_init_mulmod(rstate[5], 1022607788.0, 4294944443.0);

  yy2 = fmod(xx1 + xx2 + xx3, 4294944443.0);

  xx1 = __vectorized_cmrg_init_mulmod(rstate[3], 2824425944.0, 4294944443.0);
  xx2 = __vectorized_cmrg_init_mulmod(rstate[4],   32183930.0, 4294944443.0);
  xx3 = __vectorized_cmrg_init_mulmod(rstate[5], 2093834863.0, 4294944443.0);

  yy3 = fmod(xx1 + xx2 + xx3, 4294944443.0);

  rstate[3] = yy1;
  rstate[4] = yy2;
  rstate[5] = yy3;
}

int vectorized_cmrg_init(int num_streams, double seed)
{
  uint32_t i, j;
  if(num_streams == 0) return 0;

  vec_cmrg_num_streams = ((num_streams + NUM_ELEMENTS - 1) / NUM_ELEMENTS) * NUM_ELEMENTS;

  double rseed[6] = {seed, seed, seed, seed, seed, seed};

  posix_memalign((void**)&rstatv, ALIGN_AMOUNT, sizeof(rstatv[0]) * vec_cmrg_num_streams * 6);
  if(!rstatv) return 0;
  
  debug("Allocated %d stream states", vec_cmrg_num_streams);

  rscale_vec = vec_set1(1.0/4294967088.0);

  for(i=0; i<vec_cmrg_num_streams; i++) {
    debug("initialize stream %d", i);
    for(j=0; j<6; j++) {
      rstatv[state_idx(i, j)] = rseed[j];
    }
    __vectorized_cmrg_init_rskip(rseed);
  }

  debug("Initialized %d streams", vec_cmrg_num_streams);

  return vec_cmrg_num_streams/NUM_ELEMENTS;
}

_VEC_TYPE vectorized_cmrg_rand(int stream_group) {
  return (vectorized_cmrg_rand_group(((_VEC_TYPE*)rstatv)+vec_idx(stream_group)));
}

void vectorized_cmrg_cleanup() {
  debug("Freeing state");
  if(rstatv) free(rstatv);
}

#ifdef TESTING

#include <sys/time.h>

// returns the current time in seconds.
// used for timing the program
double getTime()
{
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+((double)t.tv_usec)/1000000.0;
}

int main(int argc, char* argv[]){
  int num_streams = 2;
  int num_to_generate = 5;
  double start;
  int group;
  int i;
  int num_stream_groups = vectorized_cmrg_init(num_streams, 12345.0);

  check(num_stream_groups, "Initialize failed");

  debug("Groups: %d", num_stream_groups);
  debug("Generating %d random numbers for each stream", num_to_generate);
 
  start = getTime();
  
  for(group=0; group<num_stream_groups; group++) {
    for(i=0; i<num_to_generate; i++) {
      _VEC_TYPE temp = vectorized_cmrg_rand(group);
      double* values = (double*)&temp;
      debug("group %d, values: %lf, %lf",group, values[0], values[1]);
    }
  }

  log_info("%lf",getTime()-start);

  vectorized_cmrg_cleanup();
  return 0;

 error:
  vectorized_cmrg_cleanup();  
  return 1;
}
#endif
