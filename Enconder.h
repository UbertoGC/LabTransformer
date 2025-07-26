#ifndef __ENCODER_H__
#define __ENCODER_H__
#include "BloqueTransformer.h"
#include "CapaEmbedding.h"
class Encoder{
private:
    CapaEmbedding* embedding;
    BloqueTransformer** bloques;
    Matriz2D<double>* encoder_salida;
    int d_modelo;
    int m_entradas;
    int v_size;
    int n_bloques;
    int n_cabezas;
    int d_feedforward;
    int d_cabezas;
public:
    Encoder(int*, int*);
    void Forward(Vector2D<int>&, Matriz2D<double>&);
    ~Encoder();
};
Encoder::Encoder(int* config_transformer, int* config_encoder){
    d_modelo = config_transformer[0];
    m_entradas = config_transformer[1];
    v_size = config_transformer[2];
    n_bloques = config_encoder[0];
    n_cabezas = config_encoder[1];
    d_feedforward = config_encoder[2];
    d_cabezas = d_modelo / n_cabezas;

    encoder_salida = nullptr;
    embedding = new CapaEmbedding(v_size, m_entradas, d_modelo);
    bloques = new BloqueTransformer*[n_bloques];
    for (int i = 0; i < n_bloques; i++){
        bloques[i] = new BloqueTransformer(d_feedforward, n_cabezas, d_modelo, 2);
    }
}
void Encoder::Forward(Vector2D<int>& entrada, Matriz2D<double>& salida){
    //Limitadores
    if(entrada.lar() > m_entradas){
        std::cout<<"ERROR: entrada inadecuada para el calculo\n";
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
    if(encoder_salida != &salida){
        encoder_salida = &salida;
    }
}
Encoder::~Encoder(){
    delete embedding;
    for (int i = 0; i < n_bloques; i++){
        delete bloques[i];
    }
    delete[] bloques;
}

#endif