#ifndef __CAPAFEEDFORWARD_H__
#define __CAPAFEEDFORWARD_H__
#include <vector>
#include <string>
#include "Matriz2D.h"
template <typename N>
class CapaFeedForward {
private:
    Matriz2D<N> *entrada_feedforward;
    Matriz2D<N> pesos1;
    Matriz2D<N> pesos2;
    Matriz2D<N> pre_relu;
    Matriz2D<N> pos_relu;
    Vector2D<N> bias1;
    Vector2D<N> bias2;
    Matriz2D<N> *salida_feedforward;
    //Varibales ADAM
    Matriz2D<N> m_p1, v_p1, m_p2, v_p2;
    Vector2D<N> m_b1, v_b1, m_b2, v_b2;
    N beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    int t_adam = 1;
    bool adam_inicializado = false;
    void IniciarAdam(Vector2D<N>&, Vector2D<N>&);
    void AdamActualizar(N&, Matriz2D<N>&, Matriz2D<N>&, Vector2D<N>&, Vector2D<N>&);
public:
    CapaFeedForward(int, int, bool = false);
    void Forward(Matriz2D<N>&, Matriz2D<N>&);
    void Aprender(Matriz2D<N>&, N&, Matriz2D<N>&);
    ~CapaFeedForward();
};
template <typename N>
CapaFeedForward<N>::CapaFeedForward(int d_ff, int d_m, bool anunciar){
    entrada_feedforward = nullptr;
    pesos1.ReSize(d_m, d_ff);
    pesos2.ReSize(d_ff, d_m);
    bias1.ReSize(d_ff);
    bias2.ReSize(d_m);
    pesos1.Random();
    pesos2.Random();
    bias1.Random();
    bias2.Random();
    salida_feedforward = nullptr;
    if(anunciar){
        std::cout << "Capa FeedForward creada" << std::endl;
    }
}
template <typename N>
void CapaFeedForward<N>::IniciarAdam(Vector2D<N>& grad_bias1, Vector2D<N>& grad_bias2){
    if (!adam_inicializado) {
        m_p1 = Matriz2D<N>(pesos1.fil(), pesos1.col());
        v_p1 = Matriz2D<N>(pesos1.fil(), pesos1.col());
        m_p2 = Matriz2D<N>(pesos2.fil(), pesos2.col());
        v_p2 = Matriz2D<N>(pesos2.fil(), pesos2.col());
        m_b1 = Vector2D<N>(grad_bias1.lar());
        v_b1 = Vector2D<N>(grad_bias1.lar());
        m_b2 = Vector2D<N>(grad_bias2.lar());
        v_b2 = Vector2D<N>(grad_bias2.lar());
        adam_inicializado = true;
    }
}
template <typename N>
void CapaFeedForward<N>::AdamActualizar(N& t_a, Matriz2D<N>& gradiente_pesos1, Matriz2D<N>& gradiente_pesos2, Vector2D<N>& grad_bias1, Vector2D<N>& grad_bias2){
    m_p1 = (m_p1 * beta1) + (gradiente_pesos1 * (1 - beta1));
    gradiente_pesos1.ElementWiseCuadrado();
    v_p1 = (v_p1 * beta2) + (gradiente_pesos1 * (1 - beta2));
    Matriz2D<N> m_hat_p1 = m_p1 / N(1 - pow(beta1, t_adam));
    Matriz2D<N> v_hat_p1 = v_p1 / N(1 - pow(beta2, t_adam));
    v_hat_p1.ElementWiseRaiz();
    pesos1 -= (m_hat_p1 / (v_hat_p1 + epsilon) * t_a);

    m_p2 = (m_p2 * beta1) + (gradiente_pesos2 * (1 - beta1));
    gradiente_pesos2.ElementWiseCuadrado();
    v_p2 = (v_p2 * beta2) + (gradiente_pesos2 * (1 - beta2));
    Matriz2D<N> m_hat_p2 = m_p2 / N(1 - pow(beta1, t_adam));
    Matriz2D<N> v_hat_p2 = v_p2 / N(1 - pow(beta2, t_adam));
    v_hat_p2.ElementWiseRaiz();
    pesos2 -= (m_hat_p2 / (v_hat_p2 + epsilon) * t_a);

    m_b1 = (m_b1 * beta1) + (grad_bias1 * (1 - beta1));
    grad_bias1.ElementWiseCuadrado();
    v_b1 = (v_b1 * beta2) + (grad_bias1 * (1 - beta2));
    Vector2D<N> m_hat_b1 = m_b1 / N(1 - pow(beta1, t_adam));
    Vector2D<N> v_hat_b1 = v_b1 / N(1 - pow(beta2, t_adam));
    v_hat_b1.ElementWiseRaiz();
    bias1 -= (m_hat_b1 / (v_hat_b1 + epsilon) * t_a);

    m_b2 = (m_b2 * beta1) + (grad_bias2 * (1 - beta1));
    grad_bias2.ElementWiseCuadrado();
    v_b2 = (v_b2 * beta2) + (grad_bias2 * (1 - beta2));
    Vector2D<N> m_hat_b2 = m_b2 / N(1 - pow(beta1, t_adam));
    Vector2D<N> v_hat_b2 = v_b2 / N(1 - pow(beta2, t_adam));
    v_hat_b2.ElementWiseRaiz();
    bias2 -= (m_hat_b2 / (v_hat_b2 + epsilon) * t_a);
}
template <typename N>
void CapaFeedForward<N>::Forward(Matriz2D<N>& entrada, Matriz2D<N>& salida){
    pre_relu = Matmul(entrada, pesos1);
    pre_relu += bias1;
    pos_relu = RELU(pre_relu);
    salida = Matmul(pos_relu, pesos2);
    salida += bias2;
    if(entrada_feedforward != &entrada){
        entrada_feedforward = &entrada;
    }
    if(salida_feedforward != &salida){
        salida_feedforward = &salida;
    }
}
template <typename N>
void CapaFeedForward<N>::Aprender(Matriz2D<N>& grad_sig, N& t_a, Matriz2D<N>& grad_feedforward){
    Matriz2D<N> grad_salida = Matmul(grad_sig, pesos2.Transpuesta());
    Matriz2D<N> gradiente_pesos2 = Matmul(pos_relu.Transpuesta(), grad_sig);
    Vector2D<N> grad_bias2 = SumarFilas(grad_sig);
    Matriz2D<N> grad_pos_relu = grad_salida * DerRELU(pre_relu);
    Matriz2D<N> gradiente_pesos1 = Matmul(entrada_feedforward->Transpuesta(), grad_pos_relu);
    Vector2D<N> grad_bias1 = SumarFilas(grad_pos_relu);
    grad_feedforward = Matmul(grad_pos_relu, pesos1.Transpuesta());
    this->IniciarAdam(grad_bias1, grad_bias2);
    this->AdamActualizar(t_a, gradiente_pesos1, gradiente_pesos2, grad_bias1, grad_bias2);
    t_adam++;
}
template <typename N>
CapaFeedForward<N>::~CapaFeedForward(){
}
#endif