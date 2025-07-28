#ifndef __CAPAEMBEDDING_H__
#define __CAPAEMBEDDING_H__
#include "Matriz2D.h"
template <typename N>
class CapaEmbedding
{
private:
    Vector2D<int>* entrada_embedding;
    Vector2D<N>* CLS;
    Matriz2D<N> pesos_embedding;
    Matriz2D<N> posicion_embedding;
    Matriz2D<N>* salida_embedding;
    int d_modelo;
    bool c_vocabulario;
    bool c_CLS;
public:
    CapaEmbedding(int, int, int, bool, bool anunciar = false);
    void Forward(Vector2D<int>&, Matriz2D<N>&);
    void Aprender(Matriz2D<N>&, N&);
    ~CapaEmbedding();
};
template <typename N>
CapaEmbedding<N>::CapaEmbedding(int v_s, int m_e, int d_m, bool c_c, bool anunciar){
    d_modelo = d_m;
    c_CLS = c_c;
    salida_embedding = nullptr;
    entrada_embedding = nullptr;
    CLS = nullptr;
    if(c_CLS){
        CLS = new Vector2D<N>(d_modelo);
    }
    if(v_s == 0){
        c_vocabulario = false;   
    }else{
        c_vocabulario = true;
        pesos_embedding.ReSize(v_s, d_m);
        pesos_embedding.Random();
    }
    posicion_embedding.ReSize(m_e, d_m);
    posicion_embedding.Random();
    if(anunciar){
        std::cout<<"Capa de Embedding creada"<<std::endl;
    }
}
template <typename N>
void CapaEmbedding<N>::Forward(Vector2D<int>& entrada, Matriz2D<N>& salida){
    salida.ReSize(entrada.lar(), d_modelo);
    for (int i = 0; i < entrada.lar(); i++){
        if(c_vocabulario){
            if(c_CLS && (entrada[i] == -1)){
                salida[i] << (*CLS);
            }else if(entrada[i] >= 0 && (entrada[i] < pesos_embedding.fil())){
                salida[i] << pesos_embedding[entrada[i]];
                salida[i] += posicion_embedding[i];
            }else{
                std::cout<<"Token ID Invalido: "<<entrada[i]<<std::endl;
            }
        }else{
            if(c_CLS && (entrada[i] == -1)){
                salida[i] << (*CLS);
            }else{
                salida[i] += posicion_embedding[i];
            }
        }
    }
    if(salida_embedding != &salida){
        salida_embedding = &salida;
    }
    if(entrada_embedding != &entrada){
        entrada_embedding = &entrada;
    }
}
template <typename N>
void CapaEmbedding<N>::Aprender(Matriz2D<N>& grad_sig, N& t_a){
    for (int i = 0; i < entrada_embedding->lar(); ++i) {
        int token = (*entrada_embedding)[i];
        for (int j = 0; j < d_modelo; ++j) {
            if (c_vocabulario) {
                if(c_CLS && (token == -1)){
                    (*CLS)[j] -= (grad_sig[i][j] * t_a);
                }else if(token >= 0 && token < pesos_embedding.fil()){
                    pesos_embedding[token][j] -= (grad_sig[i][j] * t_a);
                }else{
                    posicion_embedding[i][j] -= (grad_sig[i][j] * t_a);
                }
            }else{
                if(c_CLS && (token == -1)){
                    (*CLS)[j] -= (grad_sig[i][j] * t_a);
                }else if (token == -1){
                    std::cerr << "Token inválido (sin CLS): " << token << std::endl;
                }else{
                    posicion_embedding[i][j] -= (grad_sig[i][j] * t_a);
                }
            }
        }
    }
}
template <typename N>
CapaEmbedding<N>::~CapaEmbedding(){
}
#endif