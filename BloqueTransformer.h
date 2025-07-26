#ifndef __BLOQUE_TRANSFORMER_H__
#define __BLOQUE_TRANSFORMER_H__
#include "CapaAtencion.h"
#include "CapaFeedForward.h"
class BloqueTransformer {
private:
    CapaAtencion* atencion;
    CapaFeedForward* feedforward;
public:
    BloqueTransformer(bool = false);
    BloqueTransformer(int, int, int, int, bool = false);
    void SelfForward(Matriz2D<double>&, Matriz2D<double>&);
    void CrossForward(Matriz2D<double>&, Matriz2D<double>&, Matriz2D<double>&);
    ~BloqueTransformer();
};
BloqueTransformer::BloqueTransformer(bool anunciar){
    atencion = new CapaAtencion(12, 768, 0);
    feedforward = new CapaFeedForward(3072, 768);
    if(anunciar){
        std::cout<<"Bloque Transformer creada"<<std::endl;
    }
}
BloqueTransformer::BloqueTransformer(int d_ff, int n_c, int d_m, int t_m, bool anunciar){
    atencion = new CapaAtencion(n_c, d_m, t_m);
    feedforward = new CapaFeedForward(d_ff, d_m);
    if(anunciar){
        std::cout<<"Bloque Transformer creada"<<std::endl;
    }
}
void BloqueTransformer::SelfForward(Matriz2D<double>& entrada, Matriz2D<double>& salida){
    Matriz2D<double> salida_atencion;
    atencion->SelfForward(entrada, salida_atencion);
    salida_atencion += entrada;
    salida_atencion.NormalizarFilas();
    feedforward->Forward(salida_atencion, salida);
    salida += salida_atencion;
    salida.NormalizarFilas();
}
void BloqueTransformer::CrossForward(Matriz2D<double>& decoder_entrada, Matriz2D<double>& encoder_entrada, Matriz2D<double>& salida){
    Matriz2D<double> salida_atencion;
    atencion->CrossForward(decoder_entrada, encoder_entrada, salida_atencion);
    salida_atencion += decoder_entrada;
    salida_atencion.NormalizarFilas();
    feedforward->Forward(salida_atencion, salida);
    salida += salida_atencion;
    salida.NormalizarFilas();
}
BloqueTransformer::~BloqueTransformer() {
    delete atencion;
    delete feedforward;
}
#endif