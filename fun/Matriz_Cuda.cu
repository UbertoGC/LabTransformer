#include "Matriz2D.h"

#define BLOCK_SIZE 16

// ============================================
// Kernel: Multiplicación de matrices (float)
// ============================================
__global__ void matMulKernel(const float* A, const float* B, float* C,
                             int filasA, int colsA, int colsB) {
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float val = 0.0f;

    for (int t = 0; t < (colsA + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        if (row < filasA && (t * BLOCK_SIZE + threadIdx.x) < colsA)
            tileA[threadIdx.y][threadIdx.x] = A[row * colsA + t * BLOCK_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < colsB && (t * BLOCK_SIZE + threadIdx.y) < colsA)
            tileB[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * colsB + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
            val += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < filasA && col < colsB)
        C[row * colsB + col] = val;
    
        
}

// ============================================
// Kernel: ReLU
// ============================================
__global__ void reluKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = fmaxf(0.0f, data[idx]);
}

// ============================================
// Kernel: Softmax por filas
// ============================================
__global__ void softmaxKernel(float* A, int rows, int cols) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;
    float* rowData = A + row * cols;

    // Max
    float maxVal = -1e30f;
    for (int j = tid; j < cols; j += blockDim.x)
        maxVal = fmaxf(maxVal, rowData[j]);
    shared[tid] = maxVal;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    maxVal = shared[0];

    // Exp y suma
    float sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        rowData[j] = expf(rowData[j] - maxVal);
        sum += rowData[j];
    }
    shared[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    sum = shared[0];

    for (int j = tid; j < cols; j += blockDim.x)
        rowData[j] /= sum;
}

// ============================================
// Kernel: Normalización por filas
// ============================================
__global__ void normalizeKernel(float* A, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    float* mean = shared;
    float* var = shared + 1;

    if (threadIdx.x == 0) {
        *mean = 0.0f;
        *var = 0.0f;
    }
    __syncthreads();

    atomicAdd(mean, A[row * cols + threadIdx.x]);
    __syncthreads();

    if (threadIdx.x == 0) *mean /= cols;
    __syncthreads();

    float diff = A[row * cols + threadIdx.x] - *mean;
    atomicAdd(var, diff * diff);
    __syncthreads();

    if (threadIdx.x == 0) *var = sqrtf(*var / cols);
    __syncthreads();

    if (*var > 0)
        A[row * cols + threadIdx.x] = diff / *var;
}

// ============================================
// Wrappers CUDA en Matriz2D
// ============================================
void Matriz2D::RELU_CUDA() {
    size_t size = filas * columnas;
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, datos, size * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    reluKernel<<<blocks, threads>>>(d_data, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(datos, d_data, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
}

void Matriz2D::SoftmaxFilas_CUDA() {
    size_t size = filas * columnas;
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, datos, size * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    size_t shared_mem = threads * sizeof(float);
    softmaxKernel<<<filas, threads, shared_mem>>>(d_data, filas, columnas);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(datos, d_data, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
}

Matriz2D Matriz2D::MultiplicarCUDA(const Matriz2D& B) const {
    if (columnas != B.filas)
        throw std::invalid_argument("Dimensiones incompatibles para multiplicación");

    Matriz2D R(filas, B.columnas);

    float *d_A, *d_B, *d_C;
    size_t sizeA = filas * columnas * sizeof(float);
    size_t sizeB = B.filas * B.columnas * sizeof(float);
    size_t sizeC = filas * B.columnas * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    CUDA_CHECK(cudaMemcpy(d_A, datos, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.datos, sizeB, cudaMemcpyHostToDevice));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((B.columnas + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (filas + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, filas, columnas, B.columnas);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(R.datos, d_C, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return R;
}
__global__ void normalizeKernelGammaBeta(float* A, const float* gamma, const float* beta,
                                          int rows, int cols, float epsilon) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    float* sum = shared;
    float* sumSq = shared + 1;

    if (threadIdx.x == 0) {
        *sum = 0.0f;
        *sumSq = 0.0f;
    }
    __syncthreads();

    // Calcular suma y suma de cuadrados
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float val = A[row * cols + j];
        atomicAdd(sum, val);
        atomicAdd(sumSq, val * val);
    }
    __syncthreads();

    float mean = *sum / cols;
    float var = (*sumSq / cols) - (mean * mean);
    float invStd = rsqrtf(var + epsilon);

    // Normalización + γ y β
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        int idx = row * cols + j;
        float normVal = (A[idx] - mean) * invStd;
        A[idx] = normVal * gamma[j] + beta[j];
    }
}

void Matriz2D::NormalizarFilas_CUDA(const Matriz2D& gamma, const Matriz2D& beta) {
    if (gamma.fil() != 1 || beta.fil() != 1 || gamma.col() != columnas || beta.col() != columnas) {
        throw std::runtime_error("Dimensiones de gamma/beta no compatibles con NormalizarFilas_CUDA");
    }

    int size = filas * columnas;
    float *d_data, *d_gamma, *d_beta;

    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, columnas * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, columnas * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_data, datos, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma.Datos(), columnas * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta.Datos(), columnas * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    size_t shared_mem = 2 * sizeof(float); // sum y sumSq
    normalizeKernelGammaBeta<<<filas, threads, shared_mem>>>(d_data, d_gamma, d_beta, filas, columnas, 1e-6f);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(datos, d_data, size * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_data);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}


__global__ void KernelSumarFila(float* datos, const float* bias, int filas, int columnas) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < filas * columnas) {
        int col = idx % columnas;
        datos[idx] += bias[col];
    }
}
__global__ void KernelSumarMatrices(float* A, const float* B, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        A[idx] += B[idx];
    }
}


void Matriz2D::SumarFilaCUDA(const Matriz2D& fila) {
    if (fila.fil() != 1 && fila.fil() != filas) {
        std::cerr << "Error: La matriz no es compatible para broadcast en GPU." << std::endl;
        return;
    }

    int size = filas * columnas;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    float* d_datos;
    float* d_bias;
    CUDA_CHECK(cudaMalloc(&d_datos, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, columnas * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_datos, datos, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, fila.Datos(), columnas * sizeof(float), cudaMemcpyHostToDevice));

    KernelSumarFila<<<blocks, threads>>>(d_datos, d_bias, filas, columnas);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(datos, d_datos, size * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_datos);
    cudaFree(d_bias);
}
void Matriz2D::SumarMatrizCUDA(const Matriz2D& otra) {
    if (filas != otra.Filas() || columnas != otra.Columnas()) {
        throw std::runtime_error("Dimensiones incompatibles en SumarMatrizCUDA");
    }

    int total = filas * columnas;
    float *d_A, *d_B;

    CUDA_CHECK(cudaMalloc(&d_A, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, total * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, datos, total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, otra.Datos(), total * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    KernelSumarMatrices<<<blocks, threads>>>(d_A, d_B, total);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(datos, d_A, total * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
}

__global__ void KernelEscalar(float* datos, float escalar, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        datos[idx] *= escalar;
    }
}

void Matriz2D::EscalarCUDA(float escalar) {
    int total = filas * columnas;
    float* d_datos;
    cudaMalloc(&d_datos, total * sizeof(float));
    cudaMemcpy(d_datos, datos, total * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    KernelEscalar<<<blocks, threads>>>(d_datos, escalar, total);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(datos, d_datos, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_datos);
}
__global__ void DerSoftmaxFilasKernel(const float* softmax_output, const float* grad_sig, 
                                     float* output, int filas, int columnas) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    
    if (i < filas && j < columnas) {
        // Suma reducida (para el término sum(grad_sig * softmax_output))
        __shared__ float sum_shared;
        if (threadIdx.x == 0) {
            sum_shared = 0.0f;
            for (int k = 0; k < columnas; k++) {
                sum_shared += grad_sig[i * columnas + k] * softmax_output[i * columnas + k];
            }
        }
        __syncthreads();

        output[i * columnas + j] = softmax_output[i * columnas + j] * 
                                  (grad_sig[i * columnas + j] - sum_shared);
    }
}

void Matriz2D::DerSoftmaxFilasCUDA(const Matriz2D& grad_sig, Matriz2D& output) {
    float *d_softmax, *d_grad, *d_output;
    CUDA_CHECK(cudaMalloc(&d_softmax, filas * columnas * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad, filas * columnas * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, filas * columnas * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_softmax, datos, filas * columnas * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad, grad_sig.Datos(), filas * columnas * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blocks(filas);
    dim3 threads(columnas);
    DerSoftmaxFilasKernel<<<blocks, threads>>>(d_softmax, d_grad, d_output, filas, columnas);

    CUDA_CHECK(cudaMemcpy(output.Datos(), d_output, filas * columnas * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_softmax));
    CUDA_CHECK(cudaFree(d_grad));
    CUDA_CHECK(cudaFree(d_output));
}

__global__ void DerRELUKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

void Matriz2D::DerRELU_CUDA(const Matriz2D& input, Matriz2D& output) {
    float *d_input, *d_output;
    int size = filas * columnas;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, input.Datos(), size * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    DerRELUKernel<<<numBlocks, blockSize>>>(d_input, d_output, size);

    CUDA_CHECK(cudaMemcpy(output.Datos(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}
__global__ void CuadradoKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = data[idx] * data[idx];
}

void Matriz2D::ElementWiseCuadradoCUDA() {
    float* d_data;  // Puntero a memoria en GPU
    int size = filas * columnas;

    // 1. Reservar memoria en GPU
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));

    // 2. Copiar datos de CPU a GPU
    CUDA_CHECK(cudaMemcpy(d_data, datos, size * sizeof(float), cudaMemcpyHostToDevice));

    // 3. Configurar y lanzar el kernel
    int blockSize = 256;  // Hilos por bloque (óptimo para la mayoría de GPUs)
    int numBlocks = (size + blockSize - 1) / blockSize;  // Bloques necesarios
    CuadradoKernel<<<numBlocks, blockSize>>>(d_data, size);

    // 4. Copiar resultados de GPU a CPU
    CUDA_CHECK(cudaMemcpy(datos, d_data, size * sizeof(float), cudaMemcpyDeviceToHost));

    // 5. Liberar memoria de GPU
    CUDA_CHECK(cudaFree(d_data));
}

__global__ void RaizKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = sqrtf(data[idx]);
}
void Matriz2D::ElementWiseRaizCUDA() {
    float* d_data;  // Puntero a memoria en GPU
    int size = filas * columnas;

    // 1. Reservar memoria en GPU
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));

    // 2. Copiar datos de CPU a GPU
    CUDA_CHECK(cudaMemcpy(d_data, datos, size * sizeof(float), cudaMemcpyHostToDevice));

    // 3. Configurar y lanzar el kernel
    int blockSize = 256;  // Hilos por bloque (valor óptimo para la mayoría de GPUs)
    int numBlocks = (size + blockSize - 1) / blockSize;  // Bloques necesarios
    RaizKernel<<<numBlocks, blockSize>>>(d_data, size);

    // 4. Copiar resultados de GPU a CPU
    CUDA_CHECK(cudaMemcpy(datos, d_data, size * sizeof(float), cudaMemcpyDeviceToHost));

    // 5. Liberar memoria de GPU
    CUDA_CHECK(cudaFree(d_data));
}