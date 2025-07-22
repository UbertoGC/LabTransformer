#ifndef __TENSOR_H__
#define __TENSOR_H__
#include <random>
#include <omp.h>
#include <iostream>
#include <vector>
#include "Matriz2D.h"
template <typename T>
class Tensor {
private:
    T *datos;
    int *saltos;
    int *forma;
    int tamaño;
    int cantidad_dimensiones;
    void exp_con_dimension(std::vector<int>);
    void llenar_matriz(Matriz2D<T>&, std::vector<int>&);
public:
    Tensor();
    Tensor(const Tensor&);
    Tensor(const std::vector<int>&,  bool r_i = false);
    Tensor(T *, const std::vector<int>&);
    void randomizar();
    void zeros();
    std::vector<int> shape();
    std::vector<int> strides();
    int size();
    int shape_size();
    bool d_equal(const Tensor&);
    void re_dimensionar(const std::vector<int>&,const std::vector<int>&, std::vector<int>&);
    T& operator[](int);
    T& operator[](std::vector<int>);
    void imprimir();
    template <typename U>
    friend Tensor<U> TensorDot( Tensor<U>&, Tensor<U>&, const std::vector<int>&, const std::vector<int>&);
    Tensor operator*(T);
    Tensor operator*(const Tensor&);
    Tensor operator+(const Tensor&);
    Tensor operator-(const Tensor&);
    Tensor& operator=(const Tensor&);
    Tensor& operator*=(T);
    Tensor& operator*=(const Tensor&);
    Tensor& operator+=(const Tensor&);
    Tensor& operator-=(const Tensor&);
    ~Tensor();
};
template <typename T>
Tensor<T>::Tensor()
{
    datos = nullptr;
    saltos = nullptr;
    forma = nullptr;
    tamaño = 0;
    cantidad_dimensiones = 0;
}
template <typename T>
Tensor<T>::Tensor(const Tensor &B)
{
    this->tamaño = B.tamaño;
    this->forma = B.forma;
    this->saltos = B.saltos;
    
    this->datos = new double[this->tamaño];
    
    #pragma omp parallel for
    for (int i = 0; i < this->tamaño; ++i) {
        this->datos[i] = B.datos[i];
    }
}
template <typename T>
Tensor<T>::Tensor(const std::vector<int>& f, bool r_i)
{
    this->cantidad_dimensiones = f.size();
    this->forma = new int[f.size()];
    this->saltos = new int[f.size()];
    tamaño = 1;

    for (int i = 0; i < this->cantidad_dimensiones; ++i) {
        this->forma[i] = f[i];
        tamaño *= f[i];
    }
    for (int i = (this->cantidad_dimensiones - 1); i > -1 ; i--) {
        saltos[i] = 1;
        if(i < (this->cantidad_dimensiones - 1)){
            saltos[i] = saltos[i+1] * forma[i+1];
        }
    }
    datos = new double[tamaño];
    if(r_i) {
        this->randomizar();
    } else {
        this->zeros();
    }
}
template <typename T>
Tensor<T>::Tensor(T* data, const std::vector<int>& f)
{
    this->cantidad_dimensiones = f.size();
    this->forma = new int[f.size()];
    this->saltos = new int[f.size()];
    tamaño = 1;
    
    for (int i = 0; i < cantidad_dimensiones; i++) {
        this->forma[i] = f[i];
        tamaño *= f[i];
    }
    for (int i = (this->cantidad_dimensiones - 1); i > -1 ; i--) {
        saltos[i] = 1;
        if(i < (this->cantidad_dimensiones - 1)){
            saltos[i] = saltos[i+1] * forma[i+1];
        }
    }
    datos = new double[tamaño];
    for (int i = 0; i < tamaño; i++){
        datos[i] = data[i];
    }
    
}
template <typename T>
void Tensor<T>::llenar_matriz(Matriz2D<T>& matriz, std::vector<int>& orden_nuevo){
    int *nueva_forma = new int[this->cantidad_dimensiones];
    for (int i = 0; i < this->cantidad_dimensiones; i++){
        nueva_forma[i] = this->forma[orden_nuevo[i]];
    }
    int *nuevos_saltos = new int[this->cantidad_dimensiones];
    nuevos_saltos[this->cantidad_dimensiones - 1] = 1;
    for (int i = (this->cantidad_dimensiones - 2); i > -1 ; i--) {
        nuevos_saltos[i] = nuevos_saltos[i+1] * nueva_forma[i+1];
    }
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < this->tamaño; i++){
        int viejo_id[this->cantidad_dimensiones];
        int nuevo_id[this->cantidad_dimensiones];
        int res = i;
        for (int j = 0; j < this->cantidad_dimensiones; j++){
            viejo_id[j] = res / this->saltos[j];
            res = res % this->saltos[j];
        }
        for (int j = 0; j < this->cantidad_dimensiones; j++){
            nuevo_id[j] = viejo_id[orden_nuevo[j]];
        }
        int ind = 0;
        for (int i = 0; i < this->cantidad_dimensiones; i++){
            ind += nuevos_saltos[i] * nuevo_id[i];
        }
        int x = ind / matriz.col();
        int y = ind % matriz.col();
        matriz[x][y] = this->datos[i];
    }
}
template <typename T>
void Tensor<T>::exp_con_dimension(std::vector<int> n_f){
    int n_ta = 1;
    for (auto it:n_f){
        n_ta *= it;
    }
    if(n_ta == this->tamaño){
        delete[] this->forma;
        delete[] this->saltos;
        forma = nullptr;
        saltos = nullptr;
        this->cantidad_dimensiones = n_f.size();
        this->forma = new int[this->cantidad_dimensiones];
        for (int i = 0; i < this->cantidad_dimensiones; i++) {
            this->forma[i] = n_f[i];
        }
        this->saltos = new int[this->cantidad_dimensiones];
        for (int i = (this->cantidad_dimensiones - 1); i > -1 ; i--) {
            saltos[i] = 1;
            if(i < (this->cantidad_dimensiones - 1)){
                saltos[i] = saltos[i+1] * forma[i+1];
            }
        }
    }
}
template <typename T>
void Tensor<T>::randomizar() {
    if constexpr( std::is_same<T, double>::value == true ) {
        std::random_device rd;
        std::minstd_rand gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        #pragma omp parallel for
        for (int i = 0; i < tamaño; ++i) {
            datos[i] = dis(gen);
        }
    }
    else if constexpr( std::is_same<T, float>::value == true ) {
        std::random_device rd;
        std::minstd_rand gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        #pragma omp parallel for
        for (int i = 0; i < tamaño; ++i) {
            datos[i] = dis(gen);
        }
    }
    else if constexpr( std::is_same<T, int>::value == true ) {
        std::random_device rd;
        std::minstd_rand gen(rd());
        std::uniform_int_distribution<int> dis(-100, 100);
        #pragma omp parallel for
        for (int i = 0; i < tamaño; ++i) {
            datos[i] = dis(gen);
        }
    }
    else{
        std::cout<<"Aviso: El tipo de dato no es double, no se puede randomizar."<<std::endl;
    }
}
template <typename T>
void Tensor<T>::zeros() {
    if constexpr( std::is_same<T, double>::value == true ) {
        #pragma omp parallel for
        for (int i = 0; i < tamaño; ++i) {
            datos[i] = 0.0;
        }
    }
    else if constexpr( std::is_same<T, float>::value == true ) {
        #pragma omp parallel for
        for (int i = 0; i < tamaño; ++i) {
            datos[i] = 0.0f;
        }
    }
    else if constexpr( std::is_same<T, int>::value == true ) {
        #pragma omp parallel for
        for (int i = 0; i < tamaño; ++i) {
            datos[i] = 0;
        }
    }
    else{
        std::cout<<"Aviso: El tipo de dato no es double, no se puede inicializar en cero."<<std::endl;
    }
}
template <typename T>
std::vector<int> Tensor<T>::shape() {
    std::vector<int> shape_copy(cantidad_dimensiones);
    #pragma omp parallel for
    for (int i = 0; i < cantidad_dimensiones; ++i) {
        shape_copy[i] = forma[i];
    }
    return shape_copy;
}
template <typename T>
std::vector<int> Tensor<T>::strides() {
    std::vector<int> strides_copy(cantidad_dimensiones);
    #pragma omp parallel for
    for (int i = 0; i < cantidad_dimensiones; ++i) {
        strides_copy[i] = saltos[i];
    }
    return strides_copy;
}
template <typename T>
int Tensor<T>::size(){
    return tamaño;
}
template <typename T>
int Tensor<T>::shape_size(){
    return cantidad_dimensiones;
}
template <typename T>
bool Tensor<T>::d_equal(const Tensor<T>& B) {
    if (this->cantidad_dimensiones != B.cantidad_dimensiones) {
        return false;
    }
    for (int i = 0; i < this->cantidad_dimensiones; ++i) {
        if (this->forma[i] != B.forma[i]) {
            return false;
        }
    }
    return true;
}
template <typename T>
void Tensor<T>::re_dimensionar(const std::vector<int>& first, const std::vector<int>& second, std::vector<int>& result){
    result.push_back(1);
    result.push_back(1);
    for (auto it:first){
        result[0] *= forma[it];
    }
    for (auto it:second){
        result[1] *= forma[it];
    }
}
template <typename T>
void Tensor<T>::imprimir(){
    std::cout<<"[";
    for (int i = 0; i < this->tamaño; i++){
        if(i != 0){
            for (int j = 0; j < (this->cantidad_dimensiones - 1); j++){
                if(i % this->saltos[j] == 0){
                    std::cout<<"]";
                    if(j < this->cantidad_dimensiones - 2){
                        std::cout<<"\n";
                    }
                }
            }
        }
        for (int j = 0; j < (this->cantidad_dimensiones - 1); j++){
            if(i % this->saltos[j] == 0){
                std::cout<<"\n[";
            }
        }
        std::cout<<this->datos[i]<<", ";
        
    }
    for (int j = 0; j < (this->cantidad_dimensiones - 1); j++){
        if(this->tamaño % this->saltos[j] == 0){
            std::cout<<"]";
            if(j < this->cantidad_dimensiones - 2){
                std::cout<<"\n";
            }
        }
    }
    std::cout<<"\n]\n";
}
template <typename T>
T& Tensor<T>::operator[](int indice){
    return this->datos[indice];
}
template <typename T>
T& Tensor<T>::operator[](std::vector<int> indices){
    if(indices.size() != this->cantidad_dimensiones){
        throw std::invalid_argument("Falta indices");
    }
    for (int i = 0; i < indices.size(); i++){
        if(indices[i] >= this->forma[i] || indices[i] < 0){
            throw std::invalid_argument("Error: indice fuera de rango");
        }
    }
    int ind = 0;
    for (int i = 0; i < cantidad_dimensiones; i++){
        ind += this->saltos[i] * indices[i];
    }
    return this->datos[ind];
}
template <typename T>
Tensor<T> Tensor<T>::operator*(T e) {
    Tensor<T> result(*this);

    #pragma omp parallel for
    for (int i = 0; i < this->tamaño; ++i) {
        result.datos[i] *= e;
    }
    
    return result;
}
template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T> &B) {
    if (!this->d_equal(B)) {
        throw std::invalid_argument("Los tensores deben tener iguales dimensiones.");
    }

    int *result_tamaño = new int[this->cantidad_dimensiones];
    for (int i = 0; i < this->cantidad_dimensiones; ++i) {
        result_tamaño[i] = this->forma[i];
    }

    Tensor result(result_tamaño, this->cantidad_dimensiones);
    delete[] result_tamaño;
    
    #pragma omp parallel for
    for (int i = 0; i < this->tamaño; ++i) {
        result.datos[i] = this->datos[i] * B.datos[i];
    }
    
    return result;
}
template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T> &B) {
    if (!this->d_equal(B)) {
        throw std::invalid_argument("Los tensores deben tener iguales dimensiones.");
    }

    int *result_tamaño = new int[this->cantidad_dimensiones];
    for (int i = 0; i < this->cantidad_dimensiones; ++i) {
        result_tamaño[i] = this->forma[i];
    }

    Tensor result(result_tamaño, this->cantidad_dimensiones);
    delete[] result_tamaño;
    
    #pragma omp parallel for
    for (int i = 0; i < this->tamaño; ++i) {
        result.datos[i] = this->datos[i] + B.datos[i];
    }
    
    return result;
}
template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T> &B) {
    if (!this->d_equal(B)) {
        throw std::invalid_argument("Los tensores deben tener iguales dimensiones.");
    }

    int *result_tamaño = new int[this->cantidad_dimensiones];
    for (int i = 0; i < this->cantidad_dimensiones; ++i) {
        result_tamaño[i] = this->forma[i];
    }

    Tensor result(result_tamaño, this->cantidad_dimensiones);
    delete[] result_tamaño;
    
    #pragma omp parallel for
    for (int i = 0; i < this->tamaño; ++i) {
        result.datos[i] = this->datos[i] - B.datos[i];
    }
    
    return result;
}
template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T> &B) {
    if (this == &B) {
        return *this;
    }

    delete[] datos;
    delete[] forma;
    delete[] saltos;

    this->tamaño = B.tamaño;
    this->cantidad_dimensiones = B.cantidad_dimensiones;

    this->forma = new int[B.cantidad_dimensiones];
    this->saltos = new int[B.cantidad_dimensiones];
    
    for (int i = 0; i < B.cantidad_dimensiones; ++i) {
        this->forma[i] = B.forma[i];
        this->saltos[i] = B.saltos[i];
    }

    this->datos = new double[this->tamaño];
    
    #pragma omp parallel for
    for (int i = 0; i < this->tamaño; ++i) {
        this->datos[i] = B.datos[i];
    }
    
    return *this;
}
template <typename T>
Tensor<T>& Tensor<T>::operator*=(T e) {
    #pragma omp parallel for
    for (int i = 0; i < this->tamaño; ++i) {
        this->datos[i] *= e;
    }
    return *this;
}
template <typename T>
Tensor<T>& Tensor<T>::operator*=(const Tensor<T> &B) {
    if (!this->d_equal(B)) {
        throw std::invalid_argument("Los tensores deben tener iguales dimensiones.");
    }

    #pragma omp parallel for
    for (int i = 0; i < this->tamaño; ++i) {
        this->datos[i] *= B.datos[i];
    }
    return *this;
}
template <typename T>
Tensor<T>& Tensor<T>::operator+=(const Tensor<T> &B) {
    if (!this->d_equal(B)) {
        throw std::invalid_argument("Los tensores deben tener iguales dimensiones.");
    }

    #pragma omp parallel for
    for (int i = 0; i < this->tamaño; ++i) {
        this->datos[i] += B.datos[i];
    }
    return *this;
}
template <typename T>
Tensor<T>& Tensor<T>::operator-=(const Tensor<T> &B) {
    if (!this->d_equal(B)) {
        throw std::invalid_argument("Los tensores deben tener iguales dimensiones.");
    }

    #pragma omp parallel for
    for (int i = 0; i < this->tamaño; ++i) {
        this->datos[i] -= B.datos[i];
    }
    return *this;
}
template <typename T>
Tensor<T>::~Tensor()
{
    delete[] datos;
    delete[] forma;
    delete[] saltos;
}

template <typename U>
Tensor<U> TensorDot(Tensor<U>& A, Tensor<U>& B, const std::vector<int>& axesA, const std::vector<int>& axesB){
    for (int i = 0; i < axesA.size(); ++i) {
        if (A.forma[axesA[i]] != B.forma[axesB[i]]) {
            throw std::invalid_argument("Ejes incompatibles");
        }
    }
    std::vector<int> A_ordennuevo;
    std::vector<int> C_forma_final;
    std::vector<int> B_ordennuevo;
    for (int i = 0; i < A.cantidad_dimensiones; i++){
        bool flag = true;
        for (int j = 0; j < axesA.size(); j++){
            if(i == axesA[j]){
                flag = false;
                break;
            }
        }
        if(flag){
            A_ordennuevo.push_back(i);
            C_forma_final.push_back(A.forma[i]);
        }
    }
    for (int i = 0; i < B.cantidad_dimensiones; i++){
        bool flag = true;
        for (int j = 0; j < axesB.size(); j++){
            if(i == axesB[j]){
                flag = false;
                break;
            }
        }
        if(flag){
            B_ordennuevo.push_back(i);
            C_forma_final.push_back(B.forma[i]);
        }
    }
    std::vector<int> A_forma2D;
    std::vector<int> B_forma2D;
    
    A.re_dimensionar(A_ordennuevo, axesA, A_forma2D);
    B.re_dimensionar(axesB, B_ordennuevo, B_forma2D);

    A_ordennuevo.insert(A_ordennuevo.end(),axesA.begin(),axesA.end());
    B_ordennuevo.insert(B_ordennuevo.begin(),axesB.begin(),axesB.end());
    
    Matriz2D<U> A_matriz(A_forma2D[0],A_forma2D[1]);
    Matriz2D<U> B_matriz(B_forma2D[0],B_forma2D[1]);

    A.llenar_matriz(A_matriz,A_ordennuevo);
    B.llenar_matriz(B_matriz,B_ordennuevo);

    Matriz2D<U> C_matriz = A_matriz * B_matriz;
    Tensor<U> C(C_forma_final, false);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < C_matriz.fil(); i++){
        for (int j = 0; j < C_matriz.col(); j++){
            C.datos[i*C_matriz.col() + j] = C_matriz[i][j];
        }
    }
    return C;
}
#endif