#include <cfloat>  // Incluir para DBL_MAX
#include <stdio.h>

__global__ void Lax_Friedrichs(double *data, double dt_dx, int n, int t) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int par = n*(t%2);
    int impar = n*((t+1)%2);

    if (0 < i && i < n - 1) {
        data[i+impar] = (data[i-1+par]+data[i+1+par])*(0.5-0.25*(dt_dx)*(data[i+1+par] - data[i-1+par]));
    }
    // Condiciones de borde
    if (i == 0) {
        data[i+impar]=data[i+par+1];
    }
    if (i == n-1) {
        data[i+impar]=data[i+par-1];
    }
}

__global__ void getMaxKernel(double *arr, double *max_out, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    // El primer thread de cada bloque almacena el resultado en el arreglo result
    if (i == 0) {
        max_out[0] = arr[0];
    }
}
__global__ void maxReductionKernel(double *arr, int n) {
    extern __shared__ double sdata[];
    // Cada thread carga un elemento del arreglo en la memoria compartida
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        sdata[tid] = arr[i];
    } else {
        sdata[tid] = -DBL_MAX; // Un valor mínimo en caso de threads sin trabajo
    }
    
    __syncthreads();

    // Reducción paralela en la memoria compartida
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // El primer thread de cada bloque almacena el resultado en el arreglo result
    if (tid == 0) {
        arr[blockIdx.x] = sdata[0];
    }
}

__global__ void maxReductionIniKernel(double *arr,double *max_out, int n) {
    extern __shared__ double sdata[];

    // Cada thread carga un elemento del arreglo en la memoria compartida
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        sdata[tid] = arr[i];
    } else {
        sdata[tid] = -DBL_MAX; // Un valor mínimo en caso de threads sin trabajo
    }
    
    __syncthreads();

    // Reducción paralela en la memoria compartida
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // El primer thread de cada bloque almacena el resultado en el arreglo result
    if (tid == 0) {
        max_out[blockIdx.x] = sdata[0];
    }
}
__global__ void maxReductionKernelEnd(double *arr, double *result, int n) {
    extern __shared__ double sdata[];

    // Cada thread carga un elemento del arreglo en la memoria compartida
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        sdata[tid] = arr[i];
    } else {
        sdata[tid] = -DBL_MAX; // Un valor mínimo en caso de threads sin trabajo
    }
    
    __syncthreads();

    // Reducción paralela en la memoria compartida
    for (unsigned int s = blockDim.x / 2; s > 1; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // El primer thread global almacena el resultado en el arreglo result
    if (i == 0) {
        result[0] = sdata[1];
    }
}                    