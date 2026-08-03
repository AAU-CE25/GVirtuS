// Cuanto cuesta en la GPU un proceso CUDA que no hace NADA salvo crear su contexto.
// Si el ahorro por tenant del remoting es estado de contexto por proceso, este numero deberia
// rondar los ~460 MiB que separan a un pod nativo de un tenant remoto.
#include <cstdio>
#include <cuda_runtime.h>
int main(int argc,char**argv){
  cudaFree(0);                       // fuerza la creacion del contexto primario
  size_t libre,total; cudaMemGetInfo(&libre,&total);
  std::printf("contexto creado; libre=%zu MiB de %zu MiB\n", libre>>20, total>>20);
  int seg = (argc>1)?atoi(argv[1]):20;
  for(int i=0;i<seg;i++){ struct timespec t={1,0}; nanosleep(&t,nullptr); }
  return 0;
}
