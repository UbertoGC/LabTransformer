#ifndef __CAPAEMBEDDING_H__
#define __CAPAEMBEDDING_H__
#include "Matriz2D.h"
class CapaEmbedding
{
private:
    Matriz2D<double> pesos_embedding;
    Matriz2D<double> posicion_embedding;
    Matriz2D<double>* embedding_matriz;
    int d_modelo;
public:
    CapaEmbedding(int, int, int, bool anunciar = false);
    void Forward(Vector2D<int>&, Matriz2D<double>&);
    ~CapaEmbedding();
};
CapaEmbedding::CapaEmbedding(int v_s, int m_e, int d_m, bool anunciar){
    d_modelo = d_m;
    embedding_matriz = nullptr;
    pesos_embedding.ReSize(v_s, d_m);
    pesos_embedding.Random();
    posicion_embedding.ReSize(m_e, d_m);
    posicion_embedding.Random();
    if(anunciar){
        std::cout<<"Capa de Embedding creada"<<std::endl;
    }
}
void CapaEmbedding::Forward(Vector2D<int>& entrada, Matriz2D<double>& salida){
    salida.ReSize(entrada.lar(), d_modelo);
    for (int i = 0; i < entrada.lar(); i++){
        if (entrada[i] >= 0 && entrada[i] < pesos_embedding.fil()) {
            salida[i] << pesos_embedding[entrada[i]];
            salida[i] += posicion_embedding[i];
        } else {
            std::cout<<"Token ID Invalido: " << entrada[i]<<std::endl;
        }
    }
}
CapaEmbedding::~CapaEmbedding(){
}
#endif