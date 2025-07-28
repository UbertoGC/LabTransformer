#ifndef __CAPALINEAR_H__
#define __CAPALINEAR_H__
#include "Matriz2D.h"
class CapaLinear{
private:
    Matriz2D<double> pesos;
    Vector2D<double> bias;
    Matriz2D<double> promedio;
    Matriz2D<double>* entrada_puntero;
    Vector2D<double>* salida_puntero;
public:
    CapaLinear(int, int, bool = false);
    void Forward(Matriz2D<double>&, Vector2D<double>&);
    void Aprender(Vector2D<double>&, double&, Matriz2D<double>&);
    ~CapaLinear();
};
CapaLinear::CapaLinear(int d_m, int m_s, bool anunciar){
    pesos.ReSize(d_m, m_s);
    pesos.Random();
    promedio.ReSize(1, d_m);
    bias.ReSize(m_s);
    bias.Random();
    entrada_puntero = nullptr;
    salida_puntero = nullptr;
    if(anunciar){
        std::cout<<"Capa de Linearizacion creada"<<std::endl;
    }
}
void CapaLinear::Forward(Matriz2D<double>& entrada, Vector2D<double>& salida){
    promedio.Zero();
    for (int i = 0; i < entrada.fil(); i++){
        for (int j = 0; j < entrada.col(); j++){
            promedio[0][i] += entrada[i][j];
        }
    }
    promedio *= 1.0/double(entrada.fil());
    Matriz2D<double> resultado = Matmul(promedio, pesos);
    resultado += bias;
    salida.ReSize(pesos.col());
    for (int j = 0; j < pesos.col(); j++){
        salida[j] = resultado[0][j];
    }
    if(entrada_puntero != &entrada){
        entrada_puntero = &entrada;
    }
    if(salida_puntero != &salida){
        salida_puntero = &salida;
    }
}
void CapaLinear::Aprender(Vector2D<double>& gradiente_softmax, double& t_aprendisaje, Matriz2D<double>& gradiente_linear){
    Matriz2D<double> gradiente_pesos = Matmul(promedio.Transpuesta(), gradiente_softmax);
    Matriz2D<double> gradiente_promedio = Matmul(gradiente_softmax, pesos.Transpuesta());
    gradiente_pesos *= t_aprendisaje;
    pesos -= gradiente_pesos;
    gradiente_softmax *= t_aprendisaje;
    bias -= gradiente_softmax;
    gradiente_linear.ReSize(entrada_puntero->fil(), pesos.fil());
    for (int i = 0; i < entrada_puntero->fil(); i++){
        gradiente_linear[i] << gradiente_promedio[0];
    }
    gradiente_linear /= entrada_puntero->fil();
}
CapaLinear::~CapaLinear(){
}

#endif