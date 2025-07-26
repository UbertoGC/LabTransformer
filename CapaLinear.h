#ifndef __CAPALINEAR_H__
#define __CAPALINEAR_H__
#include "Matriz2D.h"
class CapaLinear{
private:
    Matriz2D<double> pesos;
    Vector2D<double> bias;
    Vector2D<double>* linear_vector;
public:
    CapaLinear(int, int, bool = false);
    void Forward(Matriz2D<double>&, Vector2D<double>&);
    ~CapaLinear();
};
CapaLinear::CapaLinear(int d_m, int m_s, bool anunciar){
    pesos.ReSize(d_m, m_s);
    pesos.Random();
    bias.ReSize(m_s);
    bias.Random();
    if(anunciar){
        std::cout<<"Capa de Linearizacion creada"<<std::endl;
    }
}
void CapaLinear::Forward(Matriz2D<double>& entrada, Vector2D<double>& salida){
    Matriz2D<double> promedio(1, pesos.fil());
    for (int i = 0; i < entrada.fil(); i++){
        for (int j = 0; j < entrada.col(); j++){
            promedio[0][i] += entrada[i][j];
        }
    }
    promedio *= 1.0/double(entrada.fil());
    Matriz2D<double> resultado = promedio * pesos;
    resultado += bias;
    salida.ReSize(pesos.col());
    for (int j = 0; j < pesos.col(); j++){
        salida[j] = resultado[0][j];
    }
    if(linear_vector != &salida){
        linear_vector = &salida;
    }
}
CapaLinear::~CapaLinear(){
}

#endif