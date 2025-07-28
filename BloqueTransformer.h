#ifndef __BLOQUE_TRANSFORMER_H__
#define __BLOQUE_TRANSFORMER_H__
#include "CapaAtencion.h"
#include "CapaFeedForward.h"
class BloqueTransformer {
private:
    Matriz2D<double>* entrada_bloque;
    Matriz2D<double>* entrada_encoder;
    Matriz2D<double>* entrada_decoder;
    CapaAtencion* atencion;
    Matriz2D<double> salida_atencion;
    Matriz2D<double> salida_norm_1;
    CapaFeedForward* feedforward;
    Matriz2D<double> salida_feedforward;
    Matriz2D<double>* salida_bloque;
    void AprenderFeedForward(Matriz2D<double>&, double&, Matriz2D<double>&);
public:
    BloqueTransformer(bool = false);
    BloqueTransformer(int, int, int, int, bool = false);
    void SelfForward(Matriz2D<double>&, Matriz2D<double>&);
    void CrossForward(Matriz2D<double>&, Matriz2D<double>&, Matriz2D<double>&);
    void SelfAprender(Matriz2D<double>&, double&, Matriz2D<double>&);
    void CrossAprender(Matriz2D<double>&, double&, Matriz2D<double>&, Matriz2D<double>&);
    ~BloqueTransformer();
};
BloqueTransformer::BloqueTransformer(bool anunciar){
    entrada_bloque = nullptr;
    entrada_encoder = nullptr;
    entrada_decoder = nullptr;
    atencion = new CapaAtencion(12, 768, 0);
    feedforward = new CapaFeedForward(3072, 768);
    salida_bloque = nullptr;
    if(anunciar){
        std::cout<<"Bloque Transformer creada"<<std::endl;
    }
}
BloqueTransformer::BloqueTransformer(int d_ff, int n_c, int d_m, int t_m, bool anunciar){
    entrada_bloque = nullptr;
    entrada_encoder = nullptr;
    entrada_decoder = nullptr;
    atencion = new CapaAtencion(n_c, d_m, t_m);
    feedforward = new CapaFeedForward(d_ff, d_m);
    salida_bloque = nullptr;
    if(anunciar){
        std::cout<<"Bloque Transformer creada"<<std::endl;
    }
}
void BloqueTransformer::AprenderFeedForward(Matriz2D<double>& grad_sig, double& t_a, Matriz2D<double>& grad_feedforward){
    Matriz2D<double> grad_normal_2 = DerNormalizarFilas(salida_feedforward + salida_norm_1, grad_sig);
    feedforward->Aprender(grad_normal_2, t_a, grad_feedforward);
}
void BloqueTransformer::SelfForward(Matriz2D<double>& entrada, Matriz2D<double>& salida){
    atencion->SelfForward(entrada, salida_atencion);
    salida_norm_1 = NormalizarFilas(salida_atencion + entrada);
    feedforward->Forward(salida_norm_1, salida_feedforward);
    salida = NormalizarFilas(salida_feedforward + salida_norm_1);
    if(entrada_bloque != &entrada){
        entrada_bloque = &entrada;
    }
    if(salida_bloque != &salida){
        salida_bloque = &salida;
    }
}
void BloqueTransformer::CrossForward(Matriz2D<double>& decoder_entrada, Matriz2D<double>& encoder_entrada, Matriz2D<double>& salida){
    atencion->CrossForward(decoder_entrada, encoder_entrada, salida_atencion);
    salida_norm_1 = NormalizarFilas(salida_atencion + decoder_entrada);
    feedforward->Forward(salida_norm_1, salida_feedforward);
    salida = NormalizarFilas(salida_feedforward + salida_norm_1);
    if(entrada_encoder != &encoder_entrada){
        entrada_encoder = &encoder_entrada;
    }
    if(entrada_decoder != &decoder_entrada){
        entrada_decoder = &decoder_entrada;
    }
    if(salida_bloque != &salida){
        salida_bloque = &salida;
    }
}
void BloqueTransformer::SelfAprender(Matriz2D<double>& grad_sig, double& t_a, Matriz2D<double>& grad_salida){
    Matriz2D<double> grad_feedforward;
    this->AprenderFeedForward(grad_sig, t_a, grad_feedforward);
    Matriz2D<double> grad_normal_1 = DerNormalizarFilas(salida_atencion + (*entrada_bloque), grad_feedforward);
    atencion->SelfAprender(grad_normal_1, t_a, grad_salida);
}
void BloqueTransformer::CrossAprender(Matriz2D<double>& grad_sig, double& t_a, Matriz2D<double>& grad_decoder, Matriz2D<double>& grad_encoder){
    Matriz2D<double> grad_feedforward;
    this->AprenderFeedForward(grad_sig, t_a, grad_feedforward);
    Matriz2D<double> grad_normal_1 = DerNormalizarFilas(salida_atencion + (*entrada_decoder), grad_feedforward);
    std::cout<<grad_normal_1;
    atencion->CrossAprender(grad_normal_1, t_a, grad_decoder, grad_encoder);
    std::cout<<grad_decoder;
    std::cout<<grad_encoder;
}
BloqueTransformer::~BloqueTransformer() {
    delete atencion;
    delete feedforward;
}
#endif