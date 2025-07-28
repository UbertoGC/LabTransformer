#ifndef __CAPAFEEDFORWARD_H__
#define __CAPAFEEDFORWARD_H__
#include <vector>
#include <string>
#include "Matriz2D.h"
class CapaFeedForward {
private:
    Matriz2D<double> *entrada_feedforward;
    Matriz2D<double> pesos1;
    Matriz2D<double> pesos2;
    Matriz2D<double> pre_relu;
    Matriz2D<double> pos_relu;
    Vector2D<double> bias1;
    Vector2D<double> bias2;
    Matriz2D<double> *salida_feedforward;
public:
    CapaFeedForward(int, int, bool = false);
    void Forward(Matriz2D<double>&, Matriz2D<double>&);
    void Aprender(Matriz2D<double>&, double&, Matriz2D<double>&);
    ~CapaFeedForward();
};
CapaFeedForward::CapaFeedForward(int d_ff, int d_m, bool anunciar){
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
void CapaFeedForward::Forward(Matriz2D<double>& entrada, Matriz2D<double>& salida){
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
void CapaFeedForward::Aprender(Matriz2D<double>& grad_sig, double& t_a, Matriz2D<double>& grad_feedforward){
    Matriz2D<double> grad_salida = Matmul(grad_sig, pesos2.Transpuesta());
    Matriz2D<double> gradiente_pesos2 = Matmul(pos_relu.Transpuesta(), grad_sig);
    Vector2D<double> grad_bias2 = SumarFilas(grad_sig);
    Matriz2D<double> grad_pos_relu = grad_salida * DerRELU(pre_relu);
    Matriz2D<double> gradiente_pesos1 = Matmul(entrada_feedforward->Transpuesta(), grad_pos_relu);
    Vector2D<double> grad_bias1 = SumarFilas(grad_pos_relu);
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
CapaFeedForward::~CapaFeedForward(){
}
#endif