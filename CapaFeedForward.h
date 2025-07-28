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
    gradiente_pesos1 *= t_a;
    gradiente_pesos2 *= t_a;
    grad_bias1 *= t_a;
    grad_bias2 *= t_a;
    pesos1 -= gradiente_pesos1;
    pesos2 -= gradiente_pesos2;
    bias1 -= grad_bias1;
    bias2 -= grad_bias2;
    grad_feedforward = Matmul(grad_pos_relu,pesos1.Transpuesta());
}
template <typename N>
CapaFeedForward<N>::~CapaFeedForward(){
}
#endif