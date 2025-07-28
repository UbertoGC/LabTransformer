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
    Vector2D<double> CLS;
    Vector2D<double> salida_tokenizador;
    Matriz2D<double> salida_decoder;
    Matriz2D<double> salida_encoder;
    Matriz2D<double> salida_unificador;
    Matriz2D<double>* salida_bloques;
    Vector2D<double> salida_linear;
    Vector2D<double> salida_softmax;
    int salida_transformer;
    int num_bloques;
    int dff_bloques;
    int nc_bloques;
    int d_modelo;
    int m_salidas;
    double t_aprendisaje;
public:
    Transformer(std::function<Vector2D<double>(const T&)>, double, int*, int*, int*, int*, int*);
    void Aprendizaje(T& entrada, int clase_resultado);
    int Ejecutar(T& entrada);
    ~Transformer();
};
template <typename T>
Transformer<T>::Transformer(std::function<Vector2D<double>(const T&)> conversor, double t_a, int* config_transformer, int* config_unificador, 
                            int* config_bloques, int* config_decoder, int* config_encoder){
    std::cout<<"Creando Transformer...\n";
    d_modelo = config_transformer[0];
    m_salidas = config_transformer[1];
    t_aprendisaje = t_a;
    num_bloques = config_bloques[0];
    nc_bloques = config_bloques[1];
    dff_bloques = config_bloques[2];
    tokenizador = new CapaTokenizacion<T>(conversor, config_encoder[3]);
    CLS.ReSize(m_salidas);
    CLS.Random();
    std::cout<<"Creando Decoder...\n";
    decoder = new Decoder(config_transformer, config_decoder);
    std::cout<<"Creando Encoder...\n";
    encoder = new Encoder(config_transformer, config_encoder);
    std::cout<<"Creando Unificador...\n";
    unificador = new BloqueTransformer(config_unificador[1], config_unificador[0], d_modelo, -1);
    std::cout<<"Creando Bloques Transformer...\n";
    if(num_bloques > 0){
        salida_bloques = new Matriz2D<double>[num_bloques];
        bloques = new BloqueTransformer*[num_bloques];
        for (int i = 0; i < num_bloques; i++){
            bloques[i] = new BloqueTransformer(dff_bloques, nc_bloques, d_modelo, -1);
        }
    }else{
        bloques = nullptr;
        salida_bloques = nullptr;
    }
    std::cout<<"Creando Linearizador...\n";
    linearizador = new CapaLinear(d_modelo, m_salidas);
    std::cout<<"Finalizado Construccion de Transformer\n";
}
template <typename T>
void Transformer<T>::Aprendizaje(T& entrada, int clase_resultado){
    tokenizador->Forward(entrada, salida_tokenizador);
    encoder->Forward(salida_tokenizador, salida_encoder);
    decoder->Forward(CLS, salida_decoder);
    unificador->CrossForward(salida_decoder, salida_encoder, salida_unificador);
    Matriz2D<double>* tmp_entrada = &salida_unificador;
    if(bloques){
        for (int i = 0; i < num_bloques; i++){
            bloques[i]->SelfForward((*tmp_entrada), salida_bloques[i]);
            tmp_entrada = &salida_bloques[i];
        }
    }
    linearizador->Forward((*tmp_entrada), salida_linear);
    salida_softmax = salida_linear;
    salida_softmax.Softmax();
    std::cout<<"Softmax salida: "<<salida_softmax.lar()<<std::endl;
    salida_transformer = salida_softmax.Max();
    imprimir_vector2d(salida_softmax);
    std::cout<<"Transformer salida: "<<salida_transformer<<std::endl;
    if(salida_transformer != clase_resultado){
        Vector2D<double> gradiente_softmax = salida_softmax;
        gradiente_softmax[clase_resultado] -= 1.0;
        Matriz2D<double> gradiente_linear;
        this->linearizador->Aprender(gradiente_softmax, t_aprendisaje, gradiente_linear);
        std::cout<<"Gradiente linear salida: "<<gradiente_linear.fil()<<" X "<<gradiente_linear.col()<<std::endl;
        Matriz2D<double>* grad_bloques = nullptr;
        Matriz2D<double>* tmp_grad = &gradiente_linear;
        if(bloques){
            grad_bloques = new Matriz2D<double>[num_bloques];
            for (int i = num_bloques-1; i > -1 ; i--){
                bloques[i]->SelfAprender((*tmp_grad), t_aprendisaje, grad_bloques[i]);
                tmp_grad = &grad_bloques[i];
            }
        }
        Matriz2D<double> gradiente_uni_de;
        Matriz2D<double> gradiente_uni_en;
        unificador->CrossAprender((*tmp_grad), t_aprendisaje, gradiente_uni_de, gradiente_uni_en);
        //encoder->Aprender(gradiente_uni_en);
        decoder->Aprender(gradiente_uni_de);
    }
}
template <typename T>
int Transformer<T>::Ejecutar(T& entrada){
    tokenizador->Forward(entrada, salida_tokenizador);
    encoder->Forward(salida_tokenizador, salida_encoder);
    decoder->Forward(CLS, salida_decoder);
    unificador->CrossForward(salida_decoder, salida_encoder, salida_unificador);
    Matriz2D<double>* tmp_entrada = &salida_unificador;
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
template <typename T>
Transformer<T>::~Transformer()
{
    delete tokenizador;
    delete decoder;
    delete encoder;
}

#endif