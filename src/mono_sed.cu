// ----------------------------- Funciones auxiliares -----------------------------
__device__ double fbk(double phi, double phimax, double v0, double nrz) {
    double fbk_val = 0.0;

    // Evaluación de la condición y cálculo de fbk
    if ((phi >= 0.0) && (phi <= phimax)) {
        fbk_val = phi * v0 * pow(1.0 - phi/phimax, nrz);
    }

    return fbk_val;
}

__device__ double Godunov(double Cj, double Ck, double phimax, double v0, double nrz) {
    double gd = 0.0;
    double hat_phi = phimax / (1.0 + nrz);

    if (Cj <= Ck) {
        gd = fmin(fbk(Cj, phimax, v0, nrz), fbk(Ck, phimax, v0, nrz));
    } else if ((hat_phi - Cj) * (hat_phi - Ck) < 0.0) {
        gd = fbk(hat_phi, phimax, v0, nrz);
    } else {
        gd = fmax(fbk(Cj, phimax, v0, nrz), fbk(Ck, phimax, v0, nrz));
    }

    return gd;
}

// Kernel para actualizar phi_new
__global__ void compute_phi_new(double* phi, 
                                double dt_dx, 
                                double phimin, 
                                double phimax, 
                                double v0, 
                                double nrz,
                                int n,  
                                int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 0; // i desde 1 hasta n inclusive
    int par = n*(t%2);
    int impar = n*((t+1)%2);
    if (1<i && i<n-2) {
        phi[i+impar] = phi[i + par] - dt_dx * (Godunov(phi[i + par], phi[i + 1 + par], phimax, v0, nrz) - Godunov(phi[i-1 + par], phi[i + par], phimax, v0, nrz));
    }
    // Condiciones de borde
    if (i == 0) {
        // data[i+impar]=data[i+par+1];
        phi[i+impar]=phimin;
    }
    if (i == 1) {
        // data[i+impar]=data[i+par+1];
        phi[i+impar]=phi[i + par] - dt_dx * (Godunov(phi[i + par], phi[i + 1 + par], phimax, v0, nrz));
    }
    if (i == n-2) {
        // data[i+impar]=data[i+par+1];
        phi[i+impar]=phi[i + par] - dt_dx * ( - Godunov(phi[i-1 + par], phi[i + par], phimax, v0, nrz));
    }
    if (i == n-1) {
        phi[i+impar]=phimax;
    }    
}