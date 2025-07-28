#ifndef __ENCODER_H__
#define __ENCODER_H__
#include "BloqueTransformer.h"
#include "CapaEmbedding.h"
template <typename N>
class Encoder{
private:
    CapaEmbedding<N>* embedding;
    BloqueTransformer<N>** bloques;
    Matriz2D<N>* entradas_bloques;
    Matriz2D<N>* encoder_salida;
    int d_modelo;
    int m_entradas;
    int v_size;
    int num_bloques;
    int n_cabezas;
    int d_feedforward;
    int d_cabezas;
public:
    Encoder(int*, int*);
    void Forward(Vector2D<int>&, Matriz2D<N>&);
    void Aprender(Matriz2D<N>&, N&);
    ~Encoder();
};
template <typename N>
Encoder<N>::Encoder(int* config_transformer, int* config_encoder){
    d_modelo = config_transformer[0];
    num_bloques = config_encoder[0];
    n_cabezas = config_encoder[1];
    d_feedforward = config_encoder[2];
    v_size = config_encoder[3];
    m_entradas = config_encoder[4];
    d_cabezas = d_modelo / n_cabezas;

    encoder_salida = nullptr;
    embedding = new CapaEmbedding<N>(v_size, m_entradas, d_modelo, false);
    if(num_bloques > 0){
        entradas_bloques = new Matriz2D<N>[num_bloques];
        bloques = new BloqueTransformer<N>*[num_bloques];
        for (int i = 0; i < num_bloques; i++){
            bloques[i] = new BloqueTransformer<N>(d_feedforward, n_cabezas, d_modelo, -1);
        }
    }else{
        bloques = nullptr;
        entradas_bloques = nullptr;
    }
}
template <typename N>
void Encoder<N>::Forward(Vector2D<int>& entrada, Matriz2D<N>& salida){
    //Limitadores
    if(entrada.lar() > m_entradas){
        std::cout<<"ERROR: entrada inadecuada para el calculo\n";
    }
    if(num_bloques > 0){
        embedding->Forward(entrada, entradas_bloques[0]);
        for (int i = 0; i < num_bloques - 1; i++){
            bloques[i]->SelfForward(entradas_bloques[i], entradas_bloques[i+1]);
        }
        bloques[num_bloques-1]->SelfForward(entradas_bloques[num_bloques-1], salida);
    }else{
        embedding->Forward(entrada,salida);
    }
    if(encoder_salida != &salida){
        encoder_salida = &salida;
    }
}
template <typename N>
void Encoder<N>::Aprender(Matriz2D<N>& grad_sig, N& t_a){
    Matriz2D<N>* grad_bloques = nullptr;
    Matriz2D<N>* tmp_grad = &grad_sig;
    if(num_bloques > 0){
        grad_bloques = new Matriz2D<N>[num_bloques];
        for (int i = num_bloques-1; i > -1 ; i--){
            bloques[i]->SelfAprender((*tmp_grad), t_a, grad_bloques[i]);
            tmp_grad = &grad_bloques[i];
        }
    }
    embedding->Aprender((*tmp_grad), t_a);
}
template <typename N>
Encoder<N>::~Encoder(){
    delete embedding;
    for (int i = 0; i < num_bloques; i++){
        delete bloques[i];
    }
    delete[] bloques;
}

#endif