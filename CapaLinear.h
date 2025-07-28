#ifndef __CAPALINEAR_H__
#define __CAPALINEAR_H__
#include "Matriz2D.h"
template <typename N>
class CapaLinear{
private:
    Matriz2D<N> pesos;
    Vector2D<N> bias;
    Matriz2D<N> promedio;
    Matriz2D<N>* entrada_puntero;
    Vector2D<N>* salida_puntero;
public:
    CapaLinear(int, int, bool = false);
    void Forward(Matriz2D<N>&, Vector2D<N>&);
    void Aprender(Vector2D<N>&, N&, Matriz2D<N>&);
    ~CapaLinear();
};
template <typename N>
CapaLinear<N>::CapaLinear(int d_m, int m_s, bool anunciar){
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
template <typename N>
void CapaLinear<N>::Forward(Matriz2D<N>& entrada, Vector2D<N>& salida){
    promedio.Zero();
    for (int i = 0; i < entrada.fil(); i++){
        for (int j = 0; j < entrada.col(); j++){
            promedio[0][i] += entrada[i][j];
        }
    }
    promedio *= 1.0/N(entrada.fil());
    Matriz2D<N> resultado = Matmul(promedio, pesos);
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
template <typename N>
void CapaLinear<N>::Aprender(Vector2D<N>& gradiente_softmax, N& t_aprendisaje, Matriz2D<N>& gradiente_linear){
    Matriz2D<N> gradiente_pesos = Matmul(promedio.Transpuesta(), gradiente_softmax);
    Matriz2D<N> gradiente_promedio = Matmul(gradiente_softmax, pesos.Transpuesta());
    
    gradiente_pesos *= t_aprendisaje;
    pesos -= gradiente_pesos;
    gradiente_softmax *= t_aprendisaje;
    bias -= gradiente_softmax;
    gradiente_linear.ReSize(entrada_puntero->fil(), pesos.fil());
    for (int i = 0; i < gradiente_linear.fil(); i++){
        gradiente_linear[i] << gradiente_promedio[0];
    }
    gradiente_linear /= entrada_puntero->fil();
}
template <typename N>
CapaLinear<N>::~CapaLinear(){
}

#endif