#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__
#include "Decoder.h"
#include "Enconder.h"
#include "CapaTokenizacion.h"
#include "CapaLinear.h"
template <typename T, typename N>
class Transformer{
private:
    CapaTokenizacion<T>* tokenizador;
    Decoder<N>* decoder;
    Encoder<N>* encoder;
    BloqueTransformer<N>* unificador;
    BloqueTransformer<N>** bloques;
    CapaLinear<N>* linearizador;
    Vector2D<int> decoder_entrada;
    Vector2D<int> encoder_entrada;
    Matriz2D<N> salida_decoder;
    Matriz2D<N> salida_encoder;
    Matriz2D<N> salida_unificador;
    Matriz2D<N>* salida_bloques;
    Vector2D<N> salida_linear;
    Vector2D<N> salida_softmax;
    int salida_transformer;
    int num_bloques;
    int dff_bloques;
    int n_cabezas;
    int d_modelo;
    int m_salidas;
    N t_aprendisaje;
public:
    Transformer(std::function<Vector2D<int>(const T&)>, N, int*, int*, int*, int*, int*);
    void Aprendizaje(T& entrada, int clase_resultado);
    int Ejecutar(T& entrada);
    ~Transformer();
};
template <typename T, typename N>
Transformer<T,N>::Transformer(std::function<Vector2D<int>(const T&)> conversor, N t_a, int* config_transformer, int* config_unificador, 
                            int* config_bloques, int* config_decoder, int* config_encoder){
    std::cout<<"Creando Transformer...\n";
    d_modelo = config_transformer[0];
    m_salidas = config_transformer[1];
    t_aprendisaje = t_a;
    num_bloques = config_bloques[0];
    n_cabezas = config_bloques[1];
    dff_bloques = config_bloques[2];
    tokenizador = new CapaTokenizacion<T>(conversor, config_encoder[3]);
    decoder_entrada.ReSize(m_salidas);
    for (int i = 0; i < m_salidas; i++){
        decoder_entrada[i] = i;
    }
    decoder_entrada[0] = -1;
    std::cout<<"Creando Decoder...\n";
    decoder = new Decoder<N>(config_transformer, config_decoder);
    std::cout<<"Creando Encoder...\n";
    encoder = new Encoder<N>(config_transformer, config_encoder);
    std::cout<<"Creando Unificador...\n";
    unificador = new BloqueTransformer<N>(config_unificador[1], config_unificador[0], d_modelo, -1);
    std::cout<<"Creando Bloques Transformer...\n";
    if(num_bloques > 0){
        salida_bloques = new Matriz2D<N>[num_bloques];
        bloques = new BloqueTransformer<N>*[num_bloques];
        for (int i = 0; i < num_bloques; i++){
            bloques[i] = new BloqueTransformer<N>(dff_bloques, n_cabezas, d_modelo, -1);
        }
    }else{
        bloques = nullptr;
        salida_bloques = nullptr;
    }
    std::cout<<"Creando Linearizador...\n";
    linearizador = new CapaLinear<N>(d_modelo, m_salidas);
    std::cout<<"Finalizado Construccion de Transformer\n";
}
template <typename T, typename N>
void Transformer<T,N>::Aprendizaje(T& entrada, int clase_resultado){
    tokenizador->Forward(entrada, encoder_entrada);
    encoder->Forward(encoder_entrada, salida_encoder);
    decoder->Forward(decoder_entrada, salida_decoder);
    unificador->CrossForward(salida_decoder, salida_encoder, salida_unificador);
    Matriz2D<N>* tmp_entrada = &salida_unificador;
    if(bloques){
        for (int i = 0; i < num_bloques; i++){
            bloques[i]->SelfForward((*tmp_entrada), salida_bloques[i]);
            tmp_entrada = &salida_bloques[i];
        }
    }
    linearizador->Forward((*tmp_entrada), salida_linear);
    salida_softmax = salida_linear;
    salida_softmax.Softmax();
    salida_transformer = salida_softmax.Max();
    if(salida_transformer != clase_resultado){
        Vector2D<N> gradiente_softmax = salida_softmax;
        gradiente_softmax[clase_resultado] -= 1.0;
        Matriz2D<N> gradiente_linear;
        this->linearizador->Aprender(gradiente_softmax, t_aprendisaje, gradiente_linear);
        Matriz2D<N>* grad_bloques = nullptr;
        Matriz2D<N>* tmp_grad = &gradiente_linear;
        if(bloques){
            grad_bloques = new Matriz2D<N>[num_bloques];
            for (int i = num_bloques-1; i > -1 ; i--){
                bloques[i]->SelfAprender((*tmp_grad), t_aprendisaje, grad_bloques[i]);
                tmp_grad = &grad_bloques[i];
            }
        }
        Matriz2D<N> gradiente_uni_de;
        Matriz2D<N> gradiente_uni_en;
        unificador->CrossAprender((*tmp_grad), t_aprendisaje, gradiente_uni_de, gradiente_uni_en);
        
        encoder->Aprender(gradiente_uni_en, t_aprendisaje);
        decoder->Aprender(gradiente_uni_de, t_aprendisaje);
        return;
    }
}
template <typename T, typename N>
int Transformer<T,N>::Ejecutar(T& entrada){
    tokenizador->Forward(entrada, encoder_entrada);
    encoder->Forward(encoder_entrada, salida_encoder);
    decoder->Forward(decoder_entrada, salida_decoder);
    unificador->CrossForward(salida_decoder, salida_encoder, salida_unificador);
    Matriz2D<N>* tmp_entrada = &salida_unificador;
    if(bloques){
        for (int i = 0; i < num_bloques; i++){
            bloques[i]->SelfForward((*tmp_entrada), salida_bloques[i]);
            tmp_entrada = &salida_bloques[i];
        }
    }
    linearizador->Forward((*tmp_entrada), salida_linear);
    salida_softmax = salida_linear;
    salida_softmax.Softmax();
    salida_transformer = salida_softmax.Max();
    return salida_transformer;
}
template <typename T, typename N>
Transformer<T,N>::~Transformer()
{
    delete tokenizador;
    delete decoder;
    delete encoder;
}

#endif