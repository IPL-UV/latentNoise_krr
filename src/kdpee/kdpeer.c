// wrapper for kdpee

#include "./kdpee.h"
#include <stdlib.h>

//Test using double pointers
// lets just take in a matrix disguised as a vector store it using double pointers,
// read it using double pointers, sum it and give it back

double kdpee(const double **dimrefs, const int n, const int d, double *mins, 
               double *maxs, const double zcut, int *keys);

void kdpeer(double *x, int *n, int *d, double *mins, double *maxs, double *zcut,  double *out){
  
  //double *tmp;
  
  double result;
  int numdims = *d;
  int numdatums = *n;
  const double zcut2 = zcut[0];
  int i, j;
  
  
  // matlab way
  //tmp = x;
  //const double **dimrefs = malloc(numdims * sizeof(double *));
  //dimrefs[0] = malloc(numdatums * numdims * sizeof(double));
  //for(i = 0; i < numdims; i++){
  //  dimrefs[i] = tmp + i * numdatums;
  //}
  
  int *keys = malloc(numdatums * sizeof(int));
  double **dimrefs = malloc(numdims * sizeof(double *));
  
  for(i=0; i<numdims; i++){
    dimrefs[i] = malloc(numdatums * sizeof(double));
    for(j=0; j<numdatums; j++){
      dimrefs[i][j] = x[i*numdatums+j];
    }
  }
  
  //for(j=0; j < numdatums; ++j){
  //  keys[j] = j;
  //}
  
  // That's the matlab-specific stuff done, now let's call our calc and return it
  result = kdpee(dimrefs, numdatums, numdims, mins, maxs, zcut2, keys);
  *out = result; //y[1][1];
  
  free(keys);
  free(dimrefs);
  
  return;
}
