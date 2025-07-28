#ifndef __VECTOR2D_H__
#define __VECTOR2D_H__
#include <random>
#include <omp.h>
#include <cmath>
#include <limits>
#include <iostream>
template <typename T>
class Matriz2D;
template <typename T>
class Vector2D{
private:
    T* v;
    int largo;
public:
    friend class Matriz2D<T>;
    Vector2D();
    Vector2D(Vector2D<T>&);
    Vector2D(int);
    void Zeros();
    void Random();
    int lar() const;
    void ReSize(int);
    void Softmax();
    int Max();
    T& operator[](int) const;
    Vector2D<T>& operator=(const Vector2D<T>&);
    Vector2D<T>& operator<<(const Vector2D<T>&);
    Vector2D<T>& operator+=(const Vector2D<T>&);
    Vector2D<T>& operator-=(const Vector2D<T>&);
    Vector2D<T>& operator*=(const double&);
    Vector2D<T>& operator/=(const double&);
    template <typename U>
    friend Matriz2D<U> CrearMatriz(const Vector2D<U>&, const Vector2D<U>&);
    ~Vector2D();
};
template <typename T>
Vector2D<T>::Vector2D(){
    largo = 0;
    v = nullptr;
}
template <typename T>
Vector2D<T>::Vector2D(Vector2D<T>& B){
    this->largo = B.largo;
    this->v = new T[this->largo];
    #pragma omp parallel for
    for (int i = 0; i < this->largo; i++){
        v[i] = B.v[i];
    }
}
template <typename T>
Vector2D<T>::Vector2D(int x){
    largo = x;
    this->Zeros();
}
template <typename T>
void Vector2D<T>::ReSize(int x){
    if (x != largo) {
        delete[] v;
        largo = x;
        v = new T[largo];
    }
    this->Zeros();
}
template <typename T>
void Vector2D<T>::Softmax(){
    T suma = 0;
    T datos[this->largo];
    #pragma omp parallel for
    for (int i = 0; i < this->largo; i++){
        datos[i] = exp(v[i]);
    }
    for (int i = 0; i < this->largo; i++){
        suma += datos[i];
    }
    #pragma omp parallel for
    for (int i = 0; i < this->largo; i++){
        datos[i] /= suma;
    }
}
template <typename T>
int Vector2D<T>::Max(){
    if(largo == 0){
        return -1;
    }
    int index = 0;
    for (int i = 1; i < largo; i++){
        if(v[index] < v[i]){
            index = i;
        }
    }
    return index;
}
template <typename T>
void Vector2D<T>::Zeros(){
    v = new T[largo];
    if constexpr( std::is_same<T, double>::value == true ){
        #pragma omp parallel for
        for (int i = 0; i < largo; i++) {
            v[i] = 0.0;
        }
    }
    else if constexpr( std::is_same<T, float>::value == true ){
        #pragma omp parallel for
        for (int i = 0; i < largo; i++) {
            v[i] = 0.0f;
        }
    }
    else if constexpr( std::is_same<T, int>::value == true ){
        #pragma omp parallel for
        for (int i = 0; i < largo; i++) {
            v[i] = 0;
        }
    }
    else{
        std::cout<<"Aviso: El tipo de dato no es double, no se puede randomizar."<<std::endl;
    }
}
template <typename T>
void Vector2D<T>::Random(){
    if constexpr( std::is_same<T, double>::value == true ){
        std::random_device rd;
        std::minstd_rand gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        #pragma omp parallel for
        for (int i = 0; i < largo; i++) {
            v[i] = dis(gen);
        }
    }
    else if constexpr( std::is_same<T, float>::value == true ){
        std::random_device rd;
        std::minstd_rand gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        #pragma omp parallel for
        for (int i = 0; i < largo; i++) {
            v[i] = dis(gen);
        }
    }
    else if constexpr( std::is_same<T, int>::value == true ){
        std::random_device rd;
        std::minstd_rand gen(rd());
        std::uniform_int_distribution<int> dis(-100, 100);
        #pragma omp parallel for
        for (int i = 0; i < largo; i++) {
            v[i] = dis(gen);
        }
    }
    else{
        std::cout<<"Aviso: El tipo de dato no es double, no se puede randomizar."<<std::endl;
    }
}
template <typename T>
int Vector2D<T>::lar() const{
    return largo;
}
template <typename T>
T& Vector2D<T>::operator[](int i) const{
    return v[i];
}
template <typename T>
Vector2D<T>& Vector2D<T>::operator=(const Vector2D<T>& B){
    if(this->largo != B.largo){
        if(this->v != nullptr){
            delete v;
            v = nullptr;
        }
        v = new T[B.largo];
        this->largo = B.largo;
    }
    #pragma omp parallel for
    for (int i = 0; i < this->largo; i++) {
        this->v[i] = B.v[i];
    }
    return *this;
}
template <typename T>
Vector2D<T>& Vector2D<T>::operator<<(const Vector2D<T>& B){
    if (this == &B) 
        return *this;
    int minimo = std::min(this->largo, B.largo);
    #pragma omp parallel for
    for (int i = 0; i < minimo; i++) {
        this->v[i] = B.v[i];
    }
    return *this;
}
template <typename T>
Vector2D<T>& Vector2D<T>::operator+=(const Vector2D<T>& B){
    int minimo = std::min(this->largo, B.largo);
    #pragma omp parallel for
    for (int i = 0; i < minimo; i++) {
        this->v[i] += B.v[i];
    }
    return *this;
}
template <typename T>
Vector2D<T>& Vector2D<T>::operator-=(const Vector2D<T>& B){
    int minimo = std::min(this->largo, B.largo);
    #pragma omp parallel for
    for (int i = 0; i < minimo; i++) {
        this->v[i] -= B.v[i];
    }
    return *this;
}
template <typename T>
Vector2D<T>& Vector2D<T>::operator*=(const double& scalar){
    if(std::is_same<T, int>::value == true){
        #pragma omp parallel for
        for (int i = 0; i < this->largo; i++) {
            this->v[i] = T((double(this->v[i]) * scalar) + 0.5);
        }
    }
    else{
        #pragma omp parallel for
        for (int i = 0; i < this->largo; i++) {
            this->v[i] *= scalar;
        }
    }
    return *this;
}
template <typename T>
Vector2D<T>& Vector2D<T>::operator/=(const double& scalar){
    if(std::is_same<T, int>::value == true){
        #pragma omp parallel for
        for (int i = 0; i < this->largo; i++) {
            this->v[i] = T((double(this->v[i]) / scalar) + 0.5);
        }
    }
    else{
        #pragma omp parallel for
        for (int i = 0; i < this->largo; i++) {
            this->v[i] /= scalar;
        }
    }
    return *this;
}
template <typename U>
Matriz2D<U> CrearMatriz(const Vector2D<U>& A, const Vector2D<U>& B){
    Matriz2D<U> C(A.largo, B.largo);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < A.largo; i++){
        for (int j = 0; j < B.largo; ++j){
            C[i][j] = A.v[i] * B.v[j];
        }
    }
    return C;
}
template <typename T>
Vector2D<T>::~Vector2D(){
    delete[] v;
}
#endif