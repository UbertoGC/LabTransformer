#ifndef __DECODER_H__
#define __DECODER_H__
#include "BloqueTransformer.h"
#include "CapaEmbedding.h"
class Decoder{
private:
    CapaEmbedding* embedding;
    BloqueTransformer** bloques;
    Matriz2D<double>* salida_bloques;
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
    void Forward(Vector2D<double>&, Matriz2D<double>&);
    void Aprender(Matriz2D<double>&, double&, Matriz2D<double>&);
    ~Decoder();
};
Decoder::Decoder(int* config_transformer, int* config_decoder){
    d_modelo = config_transformer[0];
    n_bloques = config_decoder[0];
    n_cabezas = config_decoder[1];
    d_feedforward = config_decoder[2];
    v_size = config_decoder[3];
    m_entradas = d_modelo;
    d_cabezas = d_modelo / n_cabezas;
    
    decoder_salida = nullptr;
    embedding = new CapaEmbedding(v_size, m_entradas, d_modelo);
    if(n_bloques > 0){
        bloques = new BloqueTransformer*[n_bloques];
        salida_bloques = new Matriz2D<double>[n_bloques];
        for (int i = 0; i < n_bloques; i++){
            bloques[i] = new BloqueTransformer(d_feedforward, n_cabezas, d_modelo, 0);
        }
    }else{
        bloques = nullptr;
        salida_bloques = nullptr;
    }
}
void Decoder::Forward(Vector2D<double>& entrada, Matriz2D<double>& salida){
    //Limitadores
    if(entrada.lar() > m_entradas){
        std::cout<<"ERROR: entrada inadecuada para el cálculo\n";
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
    if(decoder_salida != &salida){
        decoder_salida = &salida;
    }
}
void Decoder::Aprender(Matriz2D<double>& grad_sig, double& t_a, Matriz2D<double>& gradiente_decoder){
    Matriz2D<double>* grad_bloques = nullptr;
    Matriz2D<double>* tmp_grad = &grad_sig;
    if(bloques){
        grad_bloques = new Matriz2D<double>[n_bloques];
        for (int i = n_bloques-1; i > -1 ; i--){
            bloques[i]->SelfAprender((*tmp_grad), t_a, grad_bloques[i]);
            tmp_grad = &grad_bloques[i];
        }
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