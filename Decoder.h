#ifndef __DECODER_H__
#define __DECODER_H__
#include "BloqueTransformer.h"
#include "CapaEmbedding.h"
template <typename N>
class Decoder{
private:
    CapaEmbedding<N>* embedding;
    BloqueTransformer<N>** bloques;
    Matriz2D<N>* entradas_bloques;
    Matriz2D<N>* decoder_salida;
    int d_modelo;
    int m_entradas;
    int v_size;
    int num_bloques;
    int n_cabezas;
    int d_feedforward;
    int d_cabezas;
public:
    Decoder(int*, int*);
    void Forward(Vector2D<int>&, Matriz2D<N>&);
    void Aprender(Matriz2D<N>&, N&);
    ~Decoder();
};
template <typename N>
Decoder<N>::Decoder(int* config_transformer, int* config_decoder){
    d_modelo = config_transformer[0];
    m_entradas = config_transformer[1];
    num_bloques = config_decoder[0];
    n_cabezas = config_decoder[1];
    d_feedforward = config_decoder[2];
    v_size = config_decoder[3];
    d_cabezas = d_modelo / n_cabezas;
    
    decoder_salida = nullptr;
    embedding = new CapaEmbedding<N>(v_size, m_entradas, d_modelo, true);
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
void Decoder<N>::Forward(Vector2D<int>& entrada, Matriz2D<N>& salida){
    if(entrada.lar() > m_entradas){
        std::cout<<"ERROR: entrada inadecuada para el cálculo\n";
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
    if(decoder_salida != &salida){
        decoder_salida = &salida;
    }
}
template <typename N>
void Decoder<N>::Aprender(Matriz2D<N>& grad_sig, N& t_a){
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
Decoder<N>::~Decoder(){
    delete embedding;
    for (int i = 0; i < num_bloques; i++){
        delete bloques[i];
    }
    delete[] bloques;
}
#endif