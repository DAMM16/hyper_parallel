#define MAX_SPECIES 10

__device__ double vMLB( int j,
                        double *v,
                        double *d,
                        double d_f,
                        double *diam,
                        int num_especies,
                        double g,
                        double u_0,
                        double lam,
                        double phi_max) {
    double v_f;
    double min_v = v[0];
    double sum_v = 0.0;
    double vectrho[MAX_SPECIES];
    double dot[MAX_SPECIES];
    double dot2[MAX_SPECIES];
    double sum_dot = 0.0;
    double sum_dot2 = 0.0;
    double vMLB_result;

    // Calcular min_v y sum_v
    for (int i = 0; i < num_especies; i++) {
        if (v[i] < min_v) {
            min_v = v[i];
        }
        sum_v += v[i];
    }

    if (min_v > 0.0 && sum_v < phi_max) {
        v_f = pow(1.0 - sum_v, lam - 2.0);
    } else {
        v_f = 0.0;
    }

    // Calcular vectrho = d - d_f
    for (int i = 0; i < num_especies; i++) {
        vectrho[i] = d[i] - d_f;
    }

    // Calcular dot[i] = vectrho[i] * v[i]
    for (int i = 0; i < num_especies; i++) {
        dot[i] = vectrho[i] * v[i];
        sum_dot += dot[i];
    }

    // Calcular dot2[i]
    for (int i = 0; i < num_especies; i++) {
        double diam_ratio_squared = (diam[i] * diam[i]) / (diam[0] * diam[0]);
        dot2[i] = diam_ratio_squared * v[i] * (vectrho[i] - sum_dot);
        sum_dot2 += dot2[i];
    }

    double diam_ratio_j_squared = (diam[j] * diam[j]) / (diam[0] * diam[0]);
    vMLB_result = (-g * diam[0]) / (18.0 * u_0) * v_f * (diam_ratio_j_squared * (vectrho[j] - sum_dot) - sum_dot2);

    return vMLB_result;
}
///////////////////////////////////////////////////////////////////
__device__ double flujovert(int j,
                            double *phil,
                            double *phir,
                            int num_especies,
                            double *d,
                            double d_f,
                            double *diam,
                            double g,
                            double u_0,
                            double lam,
                            double phi_max) {
    double flujovert_result;
    double vMLB_phil = vMLB(j, phil, d, d_f, diam, num_especies, g, u_0, lam, phi_max);
    double vMLB_phir = vMLB(j, phir, d, d_f, diam, num_especies, g, u_0, lam, phi_max);
    double vMLB_diff = vMLB_phil - vMLB_phir;
    double sign_value;

    if (phir[j] - phil[j] > 0) {
        sign_value = 1.0;
    } else if (phir[j] - phil[j] < 0) {
        sign_value = -1.0;
    } else {
        sign_value = 0.0;
    }

    flujovert_result = 0.5 * (phil[j] * vMLB_phil + phir[j] * vMLB_phir)
                       - 0.5 * fabs(vMLB_phir) * (phir[j] - phil[j])
                       - 0.5 * phil[j] * fabs(vMLB_diff) * sign_value;

    return flujovert_result;
}
///////////////////////////////////////////////////////////////////
__global__ void update_phi(double *phi,
                           double *d,
                           double d_f,
                           double *diam,
                           double g,
                           double u_0,
                           double lam,
                           double phi_max,
                           double dt,
                           double dx,
                           int num_celdas,
                           int num_especies,
                           int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int par = num_especies*(num_celdas+1)*(t%2);
    int impar = num_especies*(num_celdas+1)*((t+1)%2);
    if (i <= num_celdas && j < num_especies) {
        int idx = j * (num_celdas + 1) + i;

        if (i == 0) {
            double phi_i[MAX_SPECIES];
            double phi_ip1[MAX_SPECIES];
            for (int k = 0; k < num_especies; k++) {
                phi_i[k] = phi[k * (num_celdas + 1) + i + par];
                phi_ip1[k] = phi[k * (num_celdas + 1) + i + 1 + par];
            }
            double flujo = flujovert(j, phi_i, phi_ip1, num_especies, d, d_f, diam, g,u_0, lam, phi_max);
            phi[idx+impar] = phi[idx+par]-(dt / dx) * flujo;
        }
        else if (i == num_celdas) {
            double phi_im1[MAX_SPECIES];
            double phi_i[MAX_SPECIES];
            for (int k = 0; k < num_especies; k++) {
                phi_im1[k] = phi[k * (num_celdas + 1) + i - 1 + par];
                phi_i[k] = phi[k * (num_celdas + 1) + i + par];
            }
            double flujo = flujovert(j, phi_im1, phi_i, num_especies, d, d_f, diam, g,u_0, lam, phi_max);
            phi[idx+impar] = phi[idx+par]-(dt / dx) * (-flujo);
        }
        else {
            double phi_im1[MAX_SPECIES];
            double phi_i[MAX_SPECIES];
            double phi_ip1[MAX_SPECIES];
            for (int k = 0; k < num_especies; k++) {
                phi_im1[k] = phi[k * (num_celdas + 1) + i - 1 + par];
                phi_i[k] = phi[k * (num_celdas + 1) + i + par];
                phi_ip1[k] = phi[k * (num_celdas + 1) + i + 1 + par];
            }
            double flujo_forward = flujovert(j, phi_i, phi_ip1, num_especies, d, d_f, diam, g,u_0, lam, phi_max);
            double flujo_backward = flujovert(j, phi_im1, phi_i, num_especies, d, d_f, diam, g,u_0, lam, phi_max);
            phi[idx+impar] = phi[idx+par]-(dt / dx) * (flujo_forward - flujo_backward);
        }
    }
}