#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__
#include "Decoder.h"
#include "Enconder.h"
#include "CapaTokenizacion.h"
#include "CapaLinear.h"
template <typename T>
class Transformer{
private:
    CapaTokenizacion<T>* tokenizador;
    Decoder* decoder;
    Encoder* encoder;
    BloqueTransformer* unificador;
    BloqueTransformer** bloques;
    CapaLinear* linearizador;
    Vector2D<int> salida_tokenizador;
    Matriz2D<double> salida_decoder;
    Matriz2D<double> salida_encoder;
    Matriz2D<double> salida_unificador;
    Matriz2D<double> salida_bloques;
    Vector2D<double> salida_transformer;
    int num_bloques;
    int dff_bloques;
    int nc_bloques;
    int v_size;
    int d_modelo;
    int m_entradas;
    int m_salidas;
public:
    Transformer(std::function<Vector2D<int>(const T&)>, int*, int*, int*, int*, int*);
    void Forward(T& entrada);
    ~Transformer();
};
template <typename T>
Transformer<T>::Transformer(std::function<Vector2D<int>(const T&)> conversor, int* config_transformer, int* config_unificador, 
                            int* config_bloques, int* config_decoder, int* config_encoder){
    std::cout<<"Creando Transformer...\n";
    d_modelo = config_transformer[0];
    m_entradas = config_transformer[1];
    v_size = config_transformer[2];
    m_salidas = config_transformer[3];
    num_bloques = config_bloques[0];
    nc_bloques = config_bloques[1];
    dff_bloques = config_bloques[2];
    tokenizador = new CapaTokenizacion<T>(conversor, v_size);
    std::cout<<"Creando Decoder...\n";
    decoder = new Decoder(config_transformer, config_decoder);
    std::cout<<"Creando Encoder...\n";
    encoder = new Encoder(config_transformer, config_encoder);
    std::cout<<"Creando Unificador...\n";
    unificador = new BloqueTransformer(config_unificador[1], config_unificador[0], d_modelo, -1);
    std::cout<<"Creando Bloques Transformer...\n";
    if(num_bloques > 0){
        bloques = new BloqueTransformer*[num_bloques];
        for (int i = 0; i < num_bloques; i++){
            bloques[i] = new BloqueTransformer(dff_bloques, nc_bloques, d_modelo, -1);
        }
    }
    std::cout<<"Creando Linearizador...\n";
    linearizador = new CapaLinear(d_modelo, m_salidas);
    std::cout<<"Finalizado Construccion de Transformer\n";
}
template <typename T>
void Transformer<T>::Forward(T& entrada){
    tokenizador->Forward(entrada, salida_tokenizador);
    std::cout<<"Token salida: "<<salida_tokenizador.lar()<<std::endl;
    Vector2D<int> prueba(24);
    for (int i = 0; i < 24; i++){
        prueba[i] = salida_tokenizador[i];
    }
    encoder->Forward(salida_tokenizador, salida_encoder);
    std::cout<<"Encoder salida: "<<salida_encoder.fil()<<" x "<<salida_encoder.col()<<std::endl;
    decoder->Forward(prueba, salida_decoder);
    std::cout<<"Decoder salida: "<<salida_decoder.fil()<<" x "<<salida_decoder.col()<<std::endl;
    unificador->CrossForward(salida_decoder, salida_encoder, salida_unificador);
    std::cout<<"Unificador salida: "<<salida_unificador.fil()<<" x "<<salida_unificador.col()<<std::endl;
    Matriz2D<double> tmp_entrada = salida_unificador;
    for (int i = 0; i < num_bloques; i++){
        bloques[i]->SelfForward(tmp_entrada, salida_bloques);
        tmp_entrada = salida_bloques;
    }
    std::cout<<"Bloques salida: "<<salida_bloques.fil()<<" x "<<salida_bloques.col()<<std::endl;
    linearizador->Forward(salida_bloques, salida_transformer);
    std::cout<<"Transformer salida: "<<salida_transformer.lar()<<std::endl;
}
template <typename T>
Transformer<T>::~Transformer()
{
    delete tokenizador;
    delete decoder;
    delete encoder;
}

#endif