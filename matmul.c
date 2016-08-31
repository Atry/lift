#include <stdio.h>
#ifdef USE_OPENCL
#include "matmul_cl.h"
#else
#include "matmul_c.h"
#endif

int
main(){
  float A[8][8];
  float B[8][8];
  struct matmul_state state = {.A = A, .B = B};

  for(int i=0; i<8; i++)
    for(int j=0; j<8; j++)
      A[i][j] = i*8+j;

  matmul(&state);

  for(int i=0; i<8; i++) {
    for(int j=0; j<8; j++)
      printf("%5.0f ", B[i][j]);
    printf("\n");
  }

  return 0;
}
