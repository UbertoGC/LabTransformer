#ifndef __CAPATOKENIZACION_H__
#define __CAPATOKENIZACION_H__
#include "Matriz2D.h"
#include <unordered_map>
#include <functional>
template <typename T>
class CapaTokenizacion{
private:
    std::function<Vector2D<int>(const T&)> convertir;
    Vector2D<int> *tokenizacion_vector;
    int v_size;
public:
    CapaTokenizacion(std::function<Vector2D<int>(const T&)>&, int v_s);
    void Forward(T&, Vector2D<int>&);
    ~CapaTokenizacion();
};
template <typename T>
CapaTokenizacion<T>::CapaTokenizacion(std::function<Vector2D<int>(const T&)>&f, int v_s){
    tokenizacion_vector = nullptr;
    convertir = f;
    v_size = v_s;
}
template <typename T>
void CapaTokenizacion<T>::Forward(T& entrada, Vector2D<int>& salida){
    salida = this->convertir(entrada);
    if(salida.lar() > v_size){
        throw std::invalid_argument("Entrada muy larga");
    }else{
        if(tokenizacion_vector != &salida){
            tokenizacion_vector = &salida;
        }
    }
}
template <typename T>
CapaTokenizacion<T>::~CapaTokenizacion(){
}
#endif