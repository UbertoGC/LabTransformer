#ifndef __ENCODER_H__
#define __ENCODER_H__
#include "BloqueTransformer.h"
#include "CapaEmbedding.h"
class Encoder{
private:
    CapaEmbedding* embedding;
    BloqueTransformer** bloques;
    Matriz2D<double>* salida_bloques;
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
    void Forward(Vector2D<double>&, Matriz2D<double>&);
    void Aprender(Matriz2D<double>&, double&, Matriz2D<double>&);
    ~Encoder();
};
Encoder::Encoder(int* config_transformer, int* config_encoder){
    d_modelo = config_transformer[0];
    n_bloques = config_encoder[0];
    n_cabezas = config_encoder[1];
    d_feedforward = config_encoder[2];
    v_size = config_encoder[3];
    m_entradas = config_encoder[4];
    d_cabezas = d_modelo / n_cabezas;

    encoder_salida = nullptr;
    embedding = new CapaEmbedding(v_size, m_entradas, d_modelo);
    if(n_bloques > 0){
        bloques = new BloqueTransformer*[n_bloques];
        salida_bloques = new Matriz2D<double>[n_bloques];
        for (int i = 0; i < n_bloques; i++){
            bloques[i] = new BloqueTransformer(d_feedforward, n_cabezas, d_modelo, -1);
        }
    }else{
        bloques = nullptr;
        salida_bloques = nullptr;
    }
}
void Encoder::Forward(Vector2D<double>& entrada, Matriz2D<double>& salida){
    //Limitadores
    if(entrada.lar() > m_entradas){
        std::cout<<"ERROR: entrada inadecuada para el calculo\n";
    }
    if(n_bloques > 0){
        Matriz2D<double> salida_embedding;
        embedding->Forward(entrada,salida_embedding);
        Matriz2D<double>* tmp = &salida_embedding;
        for (int i = 0; i < n_bloques - 1; i++){
            bloques[i]->SelfForward((*tmp), salida_bloques[i]);
            tmp = &salida_bloques[i];
        }
        bloques[n_bloques - 1]->SelfForward((*tmp),salida);
    }else{
        embedding->Forward(entrada,salida);
    }
    if(encoder_salida != &salida){
        encoder_salida = &salida;
    }
}
void Encoder::Aprender(Matriz2D<double>& grad_sig, double& t_a, Matriz2D<double>& gradiente_encoder){

}
Encoder::~Encoder(){
    delete embedding;
    for (int i = 0; i < n_bloques; i++){
        delete bloques[i];
    }
    delete[] bloques;
}

#endif