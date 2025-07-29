#ifndef __CAPATOKENIZACION_H__
#define __CAPATOKENIZACION_H__

#include "Matriz2D.h"  // 

class CapaTokenizacion {
private:
    // Buffer para tokens en GPU
    float* d_token_buffer;
    int max_filas;
    int max_columnas;
    bool gpu_available;

public:
    // Constructor/Destructor
    CapaTokenizacion(int max_f, int max_c, bool use_gpu) 
        : max_filas(max_f), max_columnas(max_c), gpu_available(use_gpu) {
        d_token_buffer = nullptr;
        if (gpu_available) {
            cudaMalloc(&d_token_buffer, max_filas * max_columnas * sizeof(float));
        }
    }

    void Forward(const char* texto, int texto_len, Matriz2D& salida) {
        // Tokenización básica ASCII
        salida.ReSize(1, texto_len);  // 1 fila, N columnas
        
        for (int i = 0; i < texto_len; ++i) {
            salida(0, i) = static_cast<int>(texto[i]);
        }

        if (gpu_available) {
            ForwardCUDA(texto, texto_len, salida);
        }
    }

    ~CapaTokenizacion() {
        if (d_token_buffer) cudaFree(d_token_buffer);
    }

    // Procesamiento (sin template, tipo concreto)
    
    
    // Versión CUDA (declaración)
    void ForwardCUDA(const char* texto, int texto_len, Matriz2D& salida);
};



#endif