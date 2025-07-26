#ifndef __DECODER_H__
#define __DECODER_H__
#include "BloqueTransformer.h"
#include "CapaEmbedding.h"
class Decoder{
private:
    CapaEmbedding* embedding;
    BloqueTransformer** bloques;
    Matriz2D<double>* decoder_salida;
    int d_modelo;
    int m_entradas;
    int v_size;
    int n_bloques;
    int n_cabezas;
    int d_feedforward;
    int d_cabezas;
public:
    Decoder(int*, int*);
    void Forward(Vector2D<int>&, Matriz2D<double>&);
    ~Decoder();
};
Decoder::Decoder(int* config_transformer, int* config_decoder){
    d_modelo = config_transformer[0];
    m_entradas = config_transformer[1];
    v_size = config_transformer[2];
    n_bloques = config_decoder[0];
    n_cabezas = config_decoder[1];
    d_feedforward = config_decoder[2];
    d_cabezas = d_modelo / n_cabezas;
    
    decoder_salida = nullptr;
    embedding = new CapaEmbedding(v_size, m_entradas, d_modelo);
    bloques = new BloqueTransformer*[n_bloques];
    for (int i = 0; i < n_bloques; i++){
        bloques[i] = new BloqueTransformer(d_feedforward, n_cabezas, d_modelo, 0);
    }
}
void Decoder::Forward(Vector2D<int>& entrada, Matriz2D<double>& salida){
    //Limitadores
    if(entrada.lar() > m_entradas){
        std::cout<<"ERROR: entrada inadecuada para el cálculo\n";
    }
    Matriz2D<double> tmp;
    //Embedding
    embedding->Forward(entrada,tmp);
    //Bloques Transformer
    for (int i = 0; i < n_bloques; i++){
        bloques[i]->SelfForward(tmp,salida);
        tmp = salida;
    }
    //Asignacion de Salida
    if(decoder_salida != &salida){
        decoder_salida = &salida;
    }
}
Decoder::~Decoder(){
    delete embedding;
    for (int i = 0; i < n_bloques; i++){
        delete bloques[i];
    }
    delete[] bloques;
}

#endif