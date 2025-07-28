#ifndef __CAPAATENCION_H__
#define __CAPAATENCION_H__
#include <vector>
#include <string>
#include "Matriz2D.h"
template <typename N>
class CapaAtencion{
private:
    Matriz2D<N>* entrada_atencion;
    Matriz2D<N>* entrada_encoder;
    Matriz2D<N>* entrada_decoder;
    Matriz2D<N> atencion_proyeccion;
    Matriz2D<N> atencion_datosconcat;
    Matriz2D<N>* capa_WQ;
    Matriz2D<N>* capa_WK;
    Matriz2D<N>* capa_WV;
    Matriz2D<N>* Q;
    Matriz2D<N>* K;
    Matriz2D<N>* V;
    Matriz2D<N>* pre_softmax;
    Matriz2D<N>* pos_softmax;
    Matriz2D<N>* salida_atencion;
    Vector2D<N> bias;
    int num_cabezas;
    int t_mascara;
    int d_modelo;
    int d_cabeza;
    void ProyeccionFinal(Matriz2D<N>&);
    void AprenderProyeccion(Matriz2D<N>&, N&);
    void AprenderQKV(Matriz2D<N>&, int, Matriz2D<N>&, Matriz2D<N>&, Matriz2D<N>&, Matriz2D<N>&);
public:
    CapaAtencion(int, int, int = 0, bool = false);
    void SelfForward(Matriz2D<N>&, Matriz2D<N>&);
    void CrossForward(Matriz2D<N>&, Matriz2D<N>&, Matriz2D<N>&);
    void SelfAprender(Matriz2D<N>&, N&, Matriz2D<N>&);
    void CrossAprender(Matriz2D<N>&, N&, Matriz2D<N>&, Matriz2D<N>&);
    ~CapaAtencion();
};
template <typename N>
CapaAtencion<N>::CapaAtencion(int n_c, int d_m, int t_m, bool anunciar) {
    entrada_atencion = nullptr;
    entrada_encoder = nullptr;
    entrada_decoder = nullptr;
    num_cabezas = n_c;
    d_modelo = d_m;
    d_cabeza = d_modelo/num_cabezas;
    t_mascara = t_m;
    capa_WQ = new Matriz2D<N>[num_cabezas];
    capa_WK = new Matriz2D<N>[num_cabezas];
    capa_WV = new Matriz2D<N>[num_cabezas];
    Q = new Matriz2D<N>[num_cabezas];
    K = new Matriz2D<N>[num_cabezas];
    V = new Matriz2D<N>[num_cabezas];
    pre_softmax = new Matriz2D<N>[num_cabezas];
    pos_softmax = new Matriz2D<N>[num_cabezas];
    for (int i = 0; i < num_cabezas; i++){
        capa_WQ[i].ReSize(d_modelo, d_cabeza);
        capa_WK[i].ReSize(d_modelo, d_cabeza);
        capa_WV[i].ReSize(d_modelo, d_cabeza);
        capa_WQ[i].Random();
        capa_WK[i].Random();
        capa_WV[i].Random();
    }
    bias.ReSize(d_modelo);
    atencion_proyeccion.ReSize(d_modelo, d_modelo);
    atencion_proyeccion.Random();
    salida_atencion = nullptr;
    if(anunciar){
        std::cout<<"Capa de Atencion creada"<<std::endl;
    }
}
template <typename N>
void CapaAtencion<N>::ProyeccionFinal(Matriz2D<N>& salida){
    salida = Matmul(atencion_datosconcat, atencion_proyeccion);
    salida += bias;
    if (salida_atencion != &salida){
        salida_atencion = &salida;
    }
}
template <typename N>
void CapaAtencion<N>::AprenderProyeccion(Matriz2D<N>& grad_sig, N& t_a){
    Matriz2D<N> gradiente_proyeccion = Matmul(atencion_datosconcat.Transpuesta(), grad_sig);
    Vector2D<N> grad_bias = SumarFilas(grad_sig);
    gradiente_proyeccion *= t_a;
    grad_bias *= t_a;
    atencion_proyeccion -= gradiente_proyeccion;
    bias -= grad_bias;
}
template <typename N>
void CapaAtencion<N>::AprenderQKV(Matriz2D<N>& grad_concat, int c, Matriz2D<N>& mascara, Matriz2D<N>& grad_Q, Matriz2D<N>& grad_K, Matriz2D<N>& grad_V){
    Matriz2D<N> grad_parte_c(c*d_cabeza, d_cabeza, grad_concat);
    Matriz2D<N> grad_pos_softmax = Matmul(grad_parte_c, V[c].Transpuesta());
    Matriz2D<N> grad_pre_softmax = DerSoftmaxFilas(pos_softmax[c], grad_pos_softmax);
    grad_V = Matmul(pos_softmax[c].Transpuesta(), grad_parte_c);
    if(t_mascara == 0){
        for (int i = 0; i < mascara.fil(); i++) {
            for (int j = 0; j < mascara.col(); j++) {
                if (mascara[i][j] < -1e8) {
                    grad_pre_softmax[i][j] = 0;
                }
            }
        }
    }
    grad_Q = Matmul(grad_pre_softmax, K[c]);
    grad_Q /= sqrt(d_cabeza);
    grad_K = Matmul(grad_pre_softmax.Transpuesta(), Q[c]);
    grad_K /= sqrt(d_cabeza);
}
template <typename N>
void CapaAtencion<N>::SelfForward(Matriz2D<N>& entrada, Matriz2D<N>& salida){
    if(entrada_atencion != &entrada){
        entrada_atencion = &entrada;
    }
    atencion_datosconcat.ReSize(entrada.fil(), d_modelo);
    Matriz2D<N> mascara(entrada.fil(), entrada.fil(), t_mascara);
    #pragma omp parallel for
    for (int c = 0; c < num_cabezas; c++) {
        Q[c] = Matmul(entrada, capa_WQ[c]);
        K[c] = Matmul(entrada, capa_WK[c]);
        V[c] = Matmul(entrada, capa_WV[c]);
        pre_softmax[c] = Matmul(Q[c], K[c].Transpuesta());
        pre_softmax[c] *= (1.0 / sqrt(d_cabeza));
        pre_softmax[c] += mascara;
        pos_softmax[c] = SoftmaxFilas(pre_softmax[c]);
        Matriz2D<N> output_cabeza = Matmul(pos_softmax[c], V[c]);
        atencion_datosconcat.CopiarMatrizDatos(0, c * d_cabeza, output_cabeza);
    }
    ProyeccionFinal(salida);
}
template <typename N>
void CapaAtencion<N>::CrossForward(Matriz2D<N>& decoder_entrada, Matriz2D<N>& encoder_entrada, Matriz2D<N>& salida){
    if(entrada_decoder != &decoder_entrada){
        entrada_decoder = &decoder_entrada;
    }
    if(entrada_encoder != &encoder_entrada){
        entrada_encoder = &encoder_entrada;
    }
    atencion_datosconcat.ReSize(decoder_entrada.fil(), d_modelo);
    #pragma omp parallel for
    for (int c = 0; c < num_cabezas; c++) {
        Q[c] = Matmul(decoder_entrada, capa_WQ[c]);
        K[c] = Matmul(encoder_entrada, capa_WK[c]);
        V[c] = Matmul(encoder_entrada, capa_WV[c]);
        pre_softmax[c] = Matmul(Q[c], K[c].Transpuesta());
        pre_softmax[c] *= (1.0 / sqrt(d_cabeza));
        pos_softmax[c] = SoftmaxFilas(pre_softmax[c]);
        Matriz2D<N> output_cabeza = Matmul(pos_softmax[c], V[c]);
        atencion_datosconcat.CopiarMatrizDatos(0, c * d_cabeza, output_cabeza);
    }
    ProyeccionFinal(salida);
}
template <typename N>
void CapaAtencion<N>::SelfAprender(Matriz2D<N>& grad_sig, N& t_a, Matriz2D<N>& grad_atencion){
    this->AprenderProyeccion(grad_sig, t_a);
    Matriz2D<N> grad_concat = Matmul(grad_sig, atencion_proyeccion.Transpuesta());
    grad_atencion.ReSize(entrada_atencion->fil(), d_modelo);
    grad_atencion.Zero();
    Matriz2D<N> mascara(entrada_atencion->fil(), entrada_atencion->fil(), t_mascara);
    #pragma omp parallel for reduction(+:grad_atencion)
    for (int c = 0; c < num_cabezas; c++) {
        Matriz2D<N> grad_V;
        Matriz2D<N> grad_Q;
        Matriz2D<N> grad_K;
        AprenderQKV(grad_concat, c, mascara, grad_Q, grad_K, grad_V);
        Matriz2D<N> grad_cabeza_i = Matmul(grad_Q, capa_WQ[c].Transpuesta()) + Matmul(grad_K, capa_WK[c].Transpuesta()) + Matmul(grad_V, capa_WV[c].Transpuesta());
        Matriz2D<N> grad_WQ = Matmul(entrada_atencion->Transpuesta(), grad_Q);
        Matriz2D<N> grad_WK = Matmul(entrada_atencion->Transpuesta(), grad_K);
        Matriz2D<N> grad_WV = Matmul(entrada_atencion->Transpuesta(), grad_V);
        grad_WQ *= t_a;
        grad_WK *= t_a;
        grad_WV *= t_a;
        capa_WQ[c] -= grad_WQ;
        capa_WK[c] -= grad_WK;
        capa_WV[c] -= grad_WV;
        grad_atencion += grad_cabeza_i;
    }
}
template <typename N>
void CapaAtencion<N>::CrossAprender(Matriz2D<N>& grad_sig, N& t_a, Matriz2D<N>& grad_decoder, Matriz2D<N>& grad_encoder){
    this->AprenderProyeccion(grad_sig, t_a);
    Matriz2D<N> grad_concat = Matmul(grad_sig, atencion_proyeccion.Transpuesta());
    grad_decoder.ReSize(entrada_decoder->fil(), d_modelo);
    grad_decoder.Zero();
    grad_encoder.ReSize(entrada_encoder->fil(), d_modelo);
    grad_encoder.Zero();
    Matriz2D<N> mascara(entrada_decoder->fil(), entrada_decoder->fil(), t_mascara);
    #pragma omp parallel for
    for (int c = 0; c < num_cabezas; c++) {
        Matriz2D<N> grad_V;
        Matriz2D<N> grad_Q;
        Matriz2D<N> grad_K;
        AprenderQKV(grad_concat, c, mascara, grad_Q, grad_K, grad_V);
        Matriz2D<N> grad_cabeza_decoder_i = Matmul(grad_Q, capa_WQ[c].Transpuesta());
        Matriz2D<N> grad_cabeza_encoder_i = Matmul(grad_K, capa_WK[c].Transpuesta()) + Matmul(grad_V, capa_WV[c].Transpuesta());
        Matriz2D<N> grad_WQ = Matmul(entrada_decoder->Transpuesta(), grad_Q);
        Matriz2D<N> grad_WK = Matmul(entrada_encoder->Transpuesta(), grad_K);
        Matriz2D<N> grad_WV = Matmul(entrada_encoder->Transpuesta(), grad_V);
        grad_WQ *= t_a;
        grad_WK *= t_a;
        grad_WV *= t_a;
        capa_WQ[c] -= grad_WQ;
        capa_WK[c] -= grad_WK;
        capa_WV[c] -= grad_WV;
        #pragma omp critical
        {
            grad_decoder += grad_cabeza_decoder_i;
            grad_encoder += grad_cabeza_encoder_i;
        }
    }
}
template <typename N>
CapaAtencion<N>::~CapaAtencion(){
    delete[] capa_WQ;
    delete[] capa_WK;
    delete[] capa_WV;
}
#endif