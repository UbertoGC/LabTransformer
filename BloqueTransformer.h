#ifndef __BLOQUE_TRANSFORMER_H__
#define __BLOQUE_TRANSFORMER_H__
#include "CapaAtencion.h"
#include "CapaFeedForward.h"
class BloqueTransformer {
private:
    CapaAtencion* atencion;
    CapaFeedForward* feedforward;
public:
    BloqueTransformer(bool anunciar = false);
    BloqueTransformer(int, int, int, int, int, bool anunciar = false);
    void Forward(Matriz2D<double>& entrada, Matriz2D<double>& salida);
    ~BloqueTransformer();
};
BloqueTransformer::BloqueTransformer(bool anunciar){
    atencion = new CapaAtencion(12, 768, 64, 0);
    feedforward = new CapaFeedForward(3072, 768);
    if(anunciar){
        std::cout<<"Bloque Transformer creada"<<std::endl;
    }
}
BloqueTransformer::BloqueTransformer(int d_ff, int n_c, int d_m, int d_c, int t_m, bool anunciar) {
    atencion = new CapaAtencion(n_c, d_m, d_c, t_m);
    feedforward = new CapaFeedForward(d_ff, d_m);
    if(anunciar){
        std::cout<<"Bloque Transformer creada"<<std::endl;
    }
}
void BloqueTransformer::Forward(Matriz2D<double>& entrada, Matriz2D<double>& salida) {
    Matriz2D<double> salida_atencion;
    
    atencion->Forward(entrada, salida_atencion);
    salida_atencion += entrada;
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