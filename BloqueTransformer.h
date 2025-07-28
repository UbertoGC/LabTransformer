#ifndef __BLOQUE_TRANSFORMER_H__
#define __BLOQUE_TRANSFORMER_H__
#include "CapaAtencion.h"
#include "CapaFeedForward.h"
template <typename N>
class BloqueTransformer {
private:
    Matriz2D<N>* entrada_bloque;
    Matriz2D<N>* entrada_encoder;
    Matriz2D<N>* entrada_decoder;
    CapaAtencion<N>* atencion;
    Matriz2D<N> salida_atencion;
    Matriz2D<N> salida_norm_1;
    CapaFeedForward<N>* feedforward;
    Matriz2D<N> salida_feedforward;
    Matriz2D<N>* salida_bloque;
    void AprenderFeedForward(Matriz2D<N>&, N&, Matriz2D<N>&);
public:
    BloqueTransformer(bool = false);
    BloqueTransformer(int, int, int, int, bool = false);
    void SelfForward(Matriz2D<N>&, Matriz2D<N>&);
    void CrossForward(Matriz2D<N>&, Matriz2D<N>&, Matriz2D<N>&);
    void SelfAprender(Matriz2D<N>&, N&, Matriz2D<N>&);
    void CrossAprender(Matriz2D<N>&, N&, Matriz2D<N>&, Matriz2D<N>&);
    ~BloqueTransformer();
};
template <typename N>
BloqueTransformer<N>::BloqueTransformer(bool anunciar){
    entrada_bloque = nullptr;
    entrada_encoder = nullptr;
    entrada_decoder = nullptr;
    atencion = new CapaAtencion<N>(12, 768, 0);
    feedforward = new CapaFeedForward<N>(3072, 768);
    salida_bloque = nullptr;
    if(anunciar){
        std::cout<<"Bloque Transformer creada"<<std::endl;
    }
}
template <typename N>
BloqueTransformer<N>::BloqueTransformer(int d_ff, int n_c, int d_m, int t_m, bool anunciar){
    entrada_bloque = nullptr;
    entrada_encoder = nullptr;
    entrada_decoder = nullptr;
    atencion = new CapaAtencion<N>(n_c, d_m, t_m);
    feedforward = new CapaFeedForward<N>(d_ff, d_m);
    salida_bloque = nullptr;
    if(anunciar){
        std::cout<<"Bloque Transformer creada"<<std::endl;
    }
}
template <typename N>
void BloqueTransformer<N>::AprenderFeedForward(Matriz2D<N>& grad_sig, N& t_a, Matriz2D<N>& grad_feedforward){
    Matriz2D<N> grad_normal_2 = DerNormalizarFilas(salida_feedforward + salida_norm_1, grad_sig);
    feedforward->Aprender(grad_normal_2, t_a, grad_feedforward);
}
template <typename N>
void BloqueTransformer<N>::SelfForward(Matriz2D<N>& entrada, Matriz2D<N>& salida){
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
template <typename N>
void BloqueTransformer<N>::CrossForward(Matriz2D<N>& decoder_entrada, Matriz2D<N>& encoder_entrada, Matriz2D<N>& salida){
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
template <typename N>
void BloqueTransformer<N>::SelfAprender(Matriz2D<N>& grad_sig, N& t_a, Matriz2D<N>& grad_salida){
    Matriz2D<N> grad_feedforward;
    this->AprenderFeedForward(grad_sig, t_a, grad_feedforward);
    Matriz2D<N> grad_normal_1 = DerNormalizarFilas(salida_atencion + (*entrada_bloque), grad_feedforward);
    atencion->SelfAprender(grad_normal_1, t_a, grad_salida);
}
template <typename N>
void BloqueTransformer<N>::CrossAprender(Matriz2D<N>& grad_sig, N& t_a, Matriz2D<N>& grad_decoder, Matriz2D<N>& grad_encoder){
    Matriz2D<N> grad_feedforward;
    this->AprenderFeedForward(grad_sig, t_a, grad_feedforward);
    Matriz2D<N> grad_normal_1 = DerNormalizarFilas(salida_atencion + (*entrada_decoder), grad_feedforward);
    atencion->CrossAprender(grad_normal_1, t_a, grad_decoder, grad_encoder);
}
template <typename N>
BloqueTransformer<N>::~BloqueTransformer() {
    delete atencion;
    delete feedforward;
}
#endif