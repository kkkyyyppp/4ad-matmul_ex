#include <stdio.h>
#include <omp.h>

void main(){
    int a=1234;
    printf("omp_get_num_procs : %d\n",omp_get_num_procs());
    //omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel
    {
        printf("omp_get_thread_num / omp_get_num_threads : %d / %d\n",omp_get_thread_num(),omp_get_num_threads());
    }
}
