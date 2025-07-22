#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__
#include "Decoder.h"
#include "Enconder.h"
#include "CapaTokenizacion.h"
template <typename T>
class Transformer{
private:
    CapaTokenizacion<T>* tokenizador;
    Decoder* decoder;
    Encoder* encoder;
    Vector2D<int> salida_tokenizador;
    Matriz2D<double> salida_decoder;
    Matriz2D<double> salida_encoder;
    int v_size;
    int d_modelo;
    int m_entradas;
public:
    Transformer(std::function<Vector2D<int>(const T&)>, int*, int*, int*);
    void Forward(T& entrada);
    ~Transformer();
};
template <typename T>
Transformer<T>::Transformer(std::function<Vector2D<int>(const T&)> conversor, int* config_transformer, int* config_decoder, int* config_encoder){
    std::cout<<"Creando Transformer...\n";
    d_modelo = config_transformer[0];
    m_entradas = config_transformer[1];
    v_size = config_transformer[2];
    tokenizador = new CapaTokenizacion<T>(conversor, v_size);
    std::cout<<"Creando Decoder...\n";
    decoder = new Decoder(config_transformer, config_decoder);
    std::cout<<"Creando Encoder...\n";
    encoder = new Encoder(config_transformer, config_encoder);
    std::cout<<"Finalizado Transformer\n";
}
template <typename T>
void Transformer<T>::Forward(T& entrada){
    tokenizador->Forward(entrada, salida_tokenizador);
    encoder->Forward(salida_tokenizador,salida_encoder);
    std::cout<<"Encoder salida: "<<salida_encoder.fil()<<" x "<<salida_encoder.col()<<std::endl; 
    decoder->Forward(salida_tokenizador,salida_decoder);
    std::cout<<"Encoder salida: "<<salida_decoder.fil()<<" x "<<salida_decoder.col()<<std::endl; 
}
template <typename T>
Transformer<T>::~Transformer()
{
    delete tokenizador;
    delete decoder;
    delete encoder;
}

#endif