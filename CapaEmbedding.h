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
    //VARIABLES ADAM
    Matriz2D<N> m_embedding, v_embedding;
    Matriz2D<N> m_posicion, v_posicion;
    Vector2D<N> m_CLS, v_CLS;
    N beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
public:
    CapaEmbedding(int, int, int, bool, bool anunciar = false);
    void Forward(Vector2D<int>&, Matriz2D<N>&);
    void Aprender(Matriz2D<N>&, N&, int&);
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
        m_CLS.ReSize(d_m);
        v_CLS.ReSize(d_m);
    }
    if(v_s == 0){
        c_vocabulario = false;   
    }else{
        c_vocabulario = true;
        m_embedding.ReSize(v_s, d_m);
        v_embedding.ReSize(v_s, d_m);
        pesos_embedding.ReSize(v_s, d_m);
        pesos_embedding.Random();
    }
    m_posicion.ReSize(m_e, d_m);
    v_posicion.ReSize(m_e, d_m);
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
void CapaEmbedding<N>::Aprender(Matriz2D<N>& grad_sig, N& t_a, int& t_adam){
    for (int i = 0; i < entrada_embedding->lar(); i++) {
        int token = (*entrada_embedding)[i];
        for (int j = 0; j < d_modelo; j++) {
            N grad = grad_sig[i][j];
            if(c_CLS && (token == -1)){
                m_CLS[j] = beta1 * m_CLS[j] + (1 - beta1) * grad;
                v_CLS[j] = beta2 * v_CLS[j] + (1 - beta2) * grad * grad;
                N m_hat = m_CLS[j] / N(1 - std::pow(beta1, t_adam));
                N v_hat = v_CLS[j] / N(1 - std::pow(beta2, t_adam));
                (*CLS)[j] -= (t_a * m_hat / N(std::sqrt(v_hat) + epsilon));
            }else if (c_vocabulario){
                if(token >= 0 && token < pesos_embedding.fil()){
                    m_embedding[token][j] = beta1 * m_embedding[token][j] + (1 - beta1) * grad;
                    v_embedding[token][j] = beta2 * v_embedding[token][j] + (1 - beta2) * grad * grad;
                    N m_hat = m_embedding[token][j] / N(1 - std::pow(beta1, t_adam));
                    N v_hat = v_embedding[token][j] / N(1 - std::pow(beta2, t_adam));
                    pesos_embedding[token][j] -= (t_a * m_hat / N(std::sqrt(v_hat) + epsilon));

                    m_posicion[i][j] = (m_posicion[i][j] * beta1) + ((1 - beta1) * grad);
                    v_posicion[i][j] = (v_posicion[i][j] * beta2) + ((1 - beta2) * grad * grad);
                    N m_hat_pos = m_posicion[i][j] / N(1 - std::pow(beta1, t_adam));
                    N v_hat_pos = v_posicion[i][j] / N(1 - std::pow(beta2, t_adam));
                    posicion_embedding[i][j] -= (t_a * m_hat_pos / N(std::sqrt(v_hat_pos) + epsilon));
                }else{
                    std::cerr << "Token inválido (sin CLS): " << token << std::endl;
                }
            }else if(token != -1){
                m_posicion[i][j] = (m_posicion[i][j] * beta1) + ((1 - beta1) * grad);
                v_posicion[i][j] = (v_posicion[i][j] * beta2) + ((1 - beta2) * grad * grad);
                N m_hat_pos = m_posicion[i][j] / N(1 - std::pow(beta1, t_adam));
                N v_hat_pos = v_posicion[i][j] / N(1 - std::pow(beta2, t_adam));
                posicion_embedding[i][j] -= (t_a * m_hat_pos / N(std::sqrt(v_hat_pos) + epsilon));
            }else{
                std::cerr << "Token inválido (sin CLS): " << token << std::endl;
            }
        }
    }
}
template <typename N>
CapaEmbedding<N>::~CapaEmbedding(){
}
#endif