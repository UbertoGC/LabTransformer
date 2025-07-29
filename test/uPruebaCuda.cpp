#include <iostream>
#include <chrono>
#include <cmath>
#include "../fun/Matriz2D.h"

using namespace std;

// Función para comparar dos matrices
bool compararMatrices(const Matriz2D& a, const Matriz2D& b, float tolerancia = 1e-5f) {
    if (a.fil() != b.fil() || a.col() != b.col()) return false;
    
    for (int i = 0; i < a.fil() * a.col(); ++i) {
        if (fabs(a.Datos()[i] - b.Datos()[i]) > tolerancia) {
            return false;
        }
    }
    return true;
}

// Función para imprimir una matriz (solo para matrices pequeñas)
void imprimirMatriz(const Matriz2D& m, const string& nombre) {
    if (m.fil() > 5 || m.col() > 5) {
        cout << nombre << " es demasiado grande para imprimir\n";
        return;
    }

    cout << nombre << " (" << m.fil() << "x" << m.col() << "):\n";
    for (int i = 0; i < m.fil(); ++i) {
        for (int j = 0; j < m.col(); ++j) {
            cout << m.Datos()[i * m.col() + j] << "\t";
        }
        cout << "\n";
    }
    cout << "\n";
}

int main() {
    cout << "=== PRUEBA DE MATRIZ2D ===\n\n";

    // Crear matrices de prueba
    Matriz2D mat(3, 3);
    Matriz2D resultado_cpu(3, 3);
    Matriz2D resultado_gpu(3, 3);

    // Llenar con datos de prueba (0.1, 0.2, ..., 0.9)
    for (int i = 0; i < mat.fil() * mat.col(); ++i) {
        mat.Datos()[i] = (i + 1) * 0.1f;
    }

    // 1. Prueba de ElementWiseCuadrado
    cout << "1. Probando ElementWiseCuadrado...\n";
    resultado_cpu = mat;
    resultado_gpu = mat;

    resultado_cpu.ElementWiseCuadradoCPU();
    resultado_gpu.ElementWiseCuadradoCUDA();

    imprimirMatriz(resultado_cpu, "CPU");
    imprimirMatriz(resultado_gpu, "GPU");

    if (compararMatrices(resultado_cpu, resultado_gpu)) {
        cout << " Resultados coinciden\n\n";
    } else {
        cout << " Resultados diferentes\n\n";
    }

    // 2. Prueba de ElementWiseRaiz (usamos los valores al cuadrado como entrada)
    cout << "2. Probando ElementWiseRaiz...\n";
    resultado_cpu.ElementWiseRaizCPU();
    resultado_gpu.ElementWiseRaizCUDA();

    imprimirMatriz(resultado_cpu, "CPU");
    imprimirMatriz(resultado_gpu, "GPU");

    if (compararMatrices(resultado_cpu, resultado_gpu)) {
        cout << " Resultados coinciden\n\n";
    } else {
        cout << " Resultados diferentes\n\n";
    }

    // 3. Prueba de DerRELU
    cout << "3. Probando DerRELU...\n";
    // Creamos una matriz con valores positivos y negativos
    float valores[] = {-1.0f, 0.5f, 0.0f, 2.0f, -0.3f, 1.0f, -2.0f, 0.0f, 3.0f};
    for (int i = 0; i < 9; ++i) {
        mat.Datos()[i] = valores[i];
    }

    mat.DerRELU_CPU(mat, resultado_cpu);
    mat.DerRELU_CUDA(mat, resultado_gpu);

    imprimirMatriz(mat, "Entrada");
    imprimirMatriz(resultado_cpu, "CPU");
    imprimirMatriz(resultado_gpu, "GPU");

    if (compararMatrices(resultado_cpu, resultado_gpu)) {
        cout << " Resultados coinciden\n\n";
    } else {
        cout << " Resultados diferentes\n\n";
    }

    // 4. Prueba de DerSoftmaxFilas
    cout << "4. Probando DerSoftmaxFilas...\n";
    Matriz2D softmax_output(2, 3);
    Matriz2D grad_sig(2, 3);
    Matriz2D output_cpu(2, 3);
    Matriz2D output_gpu(2, 3);

    // Datos de prueba para softmax
    float softmax_data[] = {0.2f, 0.5f, 0.3f, 0.6f, 0.1f, 0.3f};
    float grad_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    
    std::memcpy(softmax_output.Datos(), softmax_data, sizeof(softmax_data));
    std::memcpy(grad_sig.Datos(), grad_data, sizeof(grad_data));

    // Calcular
    softmax_output.DerSoftmaxFilas(grad_sig, output_cpu);
    softmax_output.DerSoftmaxFilasCUDA(grad_sig, output_gpu);

    imprimirMatriz(softmax_output, "Softmax Output");
    imprimirMatriz(grad_sig, "Grad Signal");
    imprimirMatriz(output_cpu, "CPU");
    imprimirMatriz(output_gpu, "GPU");

    if (compararMatrices(output_cpu, output_gpu)) {
        cout << " Resultados coinciden\n\n";
    } else {
        cout << " Resultados diferentes\n\n";
    }

    cout << "=== PRUEBAS COMPLETADAS ===\n";
    return 0;
}