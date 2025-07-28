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
    //VARIABLES ADAM
    Matriz2D<N> m_pesos;
    Matriz2D<N> v_pesos;
    Vector2D<N> m_bias;
    Vector2D<N> v_bias;
    N beta1 = 0.9;
    N beta2 = 0.999;
    N eps = 1e-8;
    int t_adam = 1;
    bool adam_inicializado = false;
    void IniciarAdam();
    void AdamActualizar(N&, Matriz2D<N>&, Vector2D<N>&);
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
void CapaLinear<N>::IniciarAdam(){
    if(!adam_inicializado){
        m_pesos = Matriz2D<N>(pesos.fil(), pesos.col());
        v_pesos = Matriz2D<N>(pesos.fil(), pesos.col());
        m_bias = Vector2D<N>(bias.lar());
        v_bias = Vector2D<N>(bias.lar());
        adam_inicializado = true;
    }
}
template <typename N>
void CapaLinear<N>::AdamActualizar(N& t_a, Matriz2D<N>& gradiente_pesos, Vector2D<N>& gradiente_softmax){
    m_pesos = (m_pesos * beta1) + (gradiente_pesos * (1 - beta1));
    gradiente_pesos.ElementWiseCuadrado();
    v_pesos = (v_pesos * beta2) + (gradiente_pesos * (1 - beta2));
    Matriz2D<N> m_hat_pesos = m_pesos / N(1 - std::pow(beta1, t_adam));
    Matriz2D<N> v_hat_pesos = v_pesos / N(1 - std::pow(beta2, t_adam));
    v_hat_pesos.ElementWiseRaiz();
    pesos -= ((m_hat_pesos / (v_hat_pesos + eps)) * t_a);

    m_bias = (m_bias * beta1) + (gradiente_softmax * (1 - beta1));
    gradiente_softmax.ElementWiseCuadrado();
    v_bias = (v_bias * beta2) + (gradiente_softmax * (1 - beta2));
    Vector2D<N> m_hat_bias = m_bias / N(1 - std::pow(beta1, t_adam));
    Vector2D<N> v_hat_bias = v_bias / N(1 - std::pow(beta2, t_adam));
    v_hat_bias.ElementWiseRaiz();
    bias -= ((m_hat_bias / (v_hat_bias + eps)) * t_a);
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
void CapaLinear<N>::Aprender(Vector2D<N>& gradiente_softmax, N& t_a, Matriz2D<N>& gradiente_linear){
    Matriz2D<N> gradiente_pesos = Matmul(promedio.Transpuesta(), gradiente_softmax);
    Matriz2D<N> gradiente_promedio = Matmul(gradiente_softmax, pesos.Transpuesta());
    gradiente_linear.ReSize(entrada_puntero->fil(), pesos.fil());
    for (int i = 0; i < gradiente_linear.fil(); i++){
        gradiente_linear[i] << gradiente_promedio[0];
    }
    gradiente_linear /= entrada_puntero->fil();
    this->IniciarAdam();
    this->AdamActualizar(t_a, gradiente_pesos, gradiente_softmax);
    t_adam++;
}
template <typename N>
CapaLinear<N>::~CapaLinear(){
}

#endif