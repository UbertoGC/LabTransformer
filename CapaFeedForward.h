#ifndef __CAPAFEEDFORWARD_H__
#define __CAPAFEEDFORWARD_H__
#include <vector>
#include <string>
#include "Matriz2D.h"
class CapaFeedForward {
private:
    Matriz2D<double> pesos1;
    Matriz2D<double> pesos2;
    Vector2D<double> bias1;
    Vector2D<double> bias2;
public:
    CapaFeedForward(int, int, bool anunciar = false);
    void Forward(Matriz2D<double>& entrada, Matriz2D<double>& salida);
    ~CapaFeedForward();
};
CapaFeedForward::CapaFeedForward(int d_ff, int d_m, bool anunciar) {
    pesos1.ReSize(d_m, d_ff);
    pesos2.ReSize(d_ff, d_m);
    bias1.ReSize(d_ff);
    bias2.ReSize(d_m);
    pesos1.Random();
    pesos2.Random();
    bias1.Random();
    bias2.Random();
    if(anunciar){
        std::cout << "Capa FeedForward creada" << std::endl;
    }
}
void CapaFeedForward::Forward(Matriz2D<double>& entrada, Matriz2D<double>& salida) {
    Matriz2D<double> salida_ff = entrada * pesos1;
    salida_ff += bias1;
    salida_ff.RELU();
    salida = salida_ff * pesos2;
    salida += bias2;
    salida.RELU();
}
CapaFeedForward::~CapaFeedForward() {
}
#endif