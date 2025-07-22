#ifndef __MATRIZ2D_H__
#define __MATRIZ2D_H__
#include <random>
#include <omp.h>
#include <cmath>
#include <limits>
#include <iostream>
void imprimir_vector(std::vector<int> imp){
    std::cout<<"[";
    for (int i = 0; i < imp.size(); i++){
        std::cout<<imp[i];
        if(i == imp.size()-1){
            std::cout<<"]\n";
        }else{
            std::cout<<", ";
        }
    }
}
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
    int lar();
    void ReSize(int);
    T& operator[](int);
    Vector2D<T>& operator=(const Vector2D<T>&);
    Vector2D<T>& operator<<(const Vector2D<T>&);
    Vector2D<T>& operator+=(const Vector2D<T>&);
    Vector2D<T>& operator*=(const double&);
    Vector2D<T>& operator/=(const double&);
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
    this->v = new int[this->largo];
    #pragma omp parallel for
    for (int i = 0; i < this->largo; ++i) {
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
void Vector2D<T>::Zeros(){
    v = new T[largo];
    if constexpr( std::is_same<T, double>::value == true ) {
        #pragma omp parallel for
        for (int i = 0; i < largo; ++i) {
            v[i] = 0.0;
        }
    }
    else if constexpr( std::is_same<T, float>::value == true ) {
        #pragma omp parallel for
        for (int i = 0; i < largo; ++i) {
            v[i] = 0.0f;
        }
    }
    else if constexpr( std::is_same<T, int>::value == true ) {
        #pragma omp parallel for
        for (int i = 0; i < largo; ++i) {
            v[i] = 0;
        }
    }
    else{
        std::cout<<"Aviso: El tipo de dato no es double, no se puede randomizar."<<std::endl;
    }
}
template <typename T>
void Vector2D<T>::Random(){
    if constexpr( std::is_same<T, double>::value == true ) {
        std::random_device rd;
        std::minstd_rand gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        #pragma omp parallel for
        for (int i = 0; i < largo; ++i) {
            v[i] = dis(gen);
        }
    }
    else if constexpr( std::is_same<T, float>::value == true ) {
        std::random_device rd;
        std::minstd_rand gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        #pragma omp parallel for
        for (int i = 0; i < largo; ++i) {
            v[i] = dis(gen);
        }
    }
    else if constexpr( std::is_same<T, int>::value == true ) {
        std::random_device rd;
        std::minstd_rand gen(rd());
        std::uniform_int_distribution<int> dis(-100, 100);
        #pragma omp parallel for
        for (int i = 0; i < largo; ++i) {
            v[i] = dis(gen);
        }
    }
    else{
        std::cout<<"Aviso: El tipo de dato no es double, no se puede randomizar."<<std::endl;
    }
}
template <typename T>
int Vector2D<T>::lar(){
    return largo;
}
template <typename T>
T& Vector2D<T>::operator[](int i){
    return v[i];
}
template <typename T>
Vector2D<T>& Vector2D<T>::operator=(const Vector2D<T>& B) {
    if(this->largo != B.largo){
        if(this->v != nullptr){
            delete v;
            v = nullptr;
        }
        v = new int[B.largo];
        this->largo = B.largo;
    }
    #pragma omp parallel for
    for (int i = 0; i < this->largo; ++i) {
        this->v[i] = B.v[i];
    }
    return *this;
}
template <typename T>
Vector2D<T>& Vector2D<T>::operator<<(const Vector2D<T>& B) {
    if (this == &B) 
        return *this;
    int minimo = std::min(this->largo, B.largo);
    #pragma omp parallel for
    for (int i = 0; i < minimo; ++i) {
        this->v[i] = B.v[i];
    }
    return *this;
}
template <typename T>
Vector2D<T>& Vector2D<T>::operator+=(const Vector2D<T>& B) {
    int minimo = std::min(this->largo, B.largo);
    #pragma omp parallel for
    for (int i = 0; i < minimo; ++i) {
        this->v[i] += B.v[i];
    }
    return *this;
}
template <typename T>
Vector2D<T>& Vector2D<T>::operator*=(const double& scalar) {
    if(std::is_same<T, int>::value == true){
        #pragma omp parallel for
        for (int i = 0; i < this->largo; ++i) {
            this->v[i] = T((double(this->v[i]) * scalar) + 0.5);
        }
    }
    else{
        #pragma omp parallel for
        for (int i = 0; i < this->largo; ++i) {
            this->v[i] *= scalar;
        }
    }
    return *this;
}
template <typename T>
Vector2D<T>& Vector2D<T>::operator/=(const double& scalar) {
    if(std::is_same<T, int>::value == true){
        #pragma omp parallel for
        for (int i = 0; i < this->largo; ++i) {
            this->v[i] = T((double(this->v[i]) / scalar) + 0.5);
        }
    }
    else{
        #pragma omp parallel for
        for (int i = 0; i < this->largo; ++i) {
            this->v[i] /= scalar;
        }
    }
    return *this;
}
template <typename T>
Vector2D<T>::~Vector2D(){
    delete[] v;
}

template <typename T>
class Matriz2D
{
private:
    int alto;
    int ancho;
    T** m;
    Vector2D<T>* vectores;
    void Inicializar(int, int);
public:
    Matriz2D();
    Matriz2D(Matriz2D<T>&);
    Matriz2D(int, int, int t = 2);
    Matriz2D(int, int, T**);
    Matriz2D<T> Transpuesta();
    void Transponer();
    void Relacionar(const Matriz2D<T>&);
    void ReSize(int, int);
    void Limpiar();
    void Zero();
    void Random();
    void NormalizarFilas();
    void SoftmaxFilas();
    void RELU();
    void CopiarMatrizDatos(int, int, const Matriz2D<T>&);
    int fil();
    int col();
    Vector2D<T>& operator[](int);
    Matriz2D<T>& operator=(const Matriz2D<T>&);
    template <typename U>
    friend Matriz2D<U> operator+(const Matriz2D<U>&, const Matriz2D<U>&);
    template <typename U>
    friend Matriz2D<U> operator*(const Matriz2D<U>&, const Matriz2D<U>&);
    Matriz2D<T>& operator*=(const T&);
    Matriz2D<T>& operator+=(const T&);
    Matriz2D<T>& operator+=(const Vector2D<T>&);
    Matriz2D<T>& operator+=(const Matriz2D<T>&);
    template <typename U>
    friend std::ostream& operator<<(std::ostream&, const Matriz2D<U>&);
    ~Matriz2D();
};
template <typename T>
Matriz2D<T>::Matriz2D(){
    vectores = nullptr;
    alto = 0;
    ancho = 0;
    m = nullptr;
}
template <typename T>
Matriz2D<T>::Matriz2D(Matriz2D<T>& B){
    this->Inicializar(B.alto, B.ancho);
    #pragma omp parallel for
    for (int i = 0; i < alto; i++){
        #pragma omp simd
        for (int j = 0; j < ancho; j++){
            m[i][j] = B.m[i][j];
        }
    }
}
template <typename T>
Matriz2D<T>::Matriz2D(int x, int y, int t){
    this->Inicializar(x, y);
    if(t == 0){
        if constexpr( std::is_same<T, double>::value == true ) {
            #pragma omp parallel for
            for (int i = 0; i < alto; i++){
                #pragma omp simd
                for (int j = 0; j < ancho; j++){
                    if(j <= i){
                        m[i][j] = 0;
                    }
                    else{
                        m[i][j] = -std::numeric_limits<double>::infinity();
                    }
                }
            }
        }
        else if constexpr( std::is_same<T, float>::value == true ) {
            #pragma omp parallel for
            for (int i = 0; i < alto; i++){
                #pragma omp simd
                for (int j = 0; j < ancho; j++){
                    if(j <= i){
                        m[i][j] = 0;
                    }
                    else{
                        m[i][j] = -std::numeric_limits<float>::infinity();
                    }
                }
            }
        }
        else if constexpr( std::is_same<T, int>::value == true ) {
            #pragma omp parallel for
            for (int i = 0; i < alto; i++){
                #pragma omp simd
                for (int j = 0; j < ancho; j++){
                    if(j <= i){
                        m[i][j] = 0;
                    }
                    else{
                        m[i][j] = -std::numeric_limits<int>::infinity();
                    }
                }
            }
        }
        else{
            std::cout<<"Aviso: El tipo de dato no es double, no se puede randomizar."<<std::endl;
        }
        
    }
    else if(t == 1){
        this->Random();
    }
    else{
        this->Zero();
    }
}
template <typename T>
Matriz2D<T>::Matriz2D(int x, int y, T**data)
{
    this->Inicializar(x,y);
    #pragma omp parallel for
    for (int i = 0; i < alto; i++){
        #pragma omp simd
        for (int j = 0; j < ancho; j++){
            m[i][j] = data[i][j];
        }
    }
}
template <typename T>
void Matriz2D<T>::Inicializar(int x, int y){
    this->alto = x;
    this->ancho = y;
    this->m = new T*[alto];
    this->vectores = new Vector2D<T>[alto];
    int i = 0;
    int i_sig = i + 32;
    while (i < alto){
        #pragma omp parallel for
        for (int k = i; k < i_sig && k < alto; k++){
            this->m[k] = new T[ancho];
            this->vectores[k].v = this->m[k];
            this->vectores[k].largo = this->ancho;
        }
        i = i_sig;
        i_sig += 32;
    }
}
template <typename T>
void Matriz2D<T>::Zero(){
    if constexpr( std::is_same<T, double>::value == true ) {
        #pragma omp parallel for
        for (int i = 0; i < alto; i++){
            #pragma omp simd
            for (int j = 0; j < ancho; j++){
                m[i][j] = 0.0;
            }
        }
    }
    else if constexpr( std::is_same<T, float>::value == true ) {
        #pragma omp parallel for
        for (int i = 0; i < alto; i++){
            #pragma omp simd
            for (int j = 0; j < ancho; j++){
                m[i][j] = 0.0f;
            }
        }
    }
    else if constexpr( std::is_same<T, int>::value == true ) {
        #pragma omp parallel for
        for (int i = 0; i < alto; i++){
            #pragma omp simd
            for (int j = 0; j < ancho; j++){
                m[i][j] = 0;
            }
        }
    }
    else{
        std::cout<<"Aviso: El tipo de dato no es double, no se puede randomizar."<<std::endl;
    }
    
}
template <typename T>
Matriz2D<T> Matriz2D<T>::Transpuesta(){
    Matriz2D t(ancho, alto);
    #pragma omp parallel for
    for (int i = 0; i < alto; i++){
        #pragma omp simd
        for (int j = 0; j < ancho; j++){
            t[j][i] = m[i][j];
        }
    }
    return t;
}
template <typename T>
void Matriz2D<T>::Transponer(){
    T **tmp = m;
    m = nullptr;
    #pragma omp parallel for
    for (int i = 0; i < ancho; i++){
        vectores[i].v = nullptr;
    }
    delete[] vectores;
    this->Inicializar(ancho, alto);
    #pragma omp parallel for
    for (int i = 0; i < alto; i++){
        #pragma omp simd
        for (int j = 0; j < ancho; j++){
            m[i][j] = tmp[j][i];
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < alto; i++){
        delete[] tmp[i];
    }
    delete[] vectores;
    delete[] tmp;
}
template <typename T>
void Matriz2D<T>::Relacionar(const Matriz2D<T>& B){
    this->Limpiar();
    this->alto = B.alto;
    this->ancho = B.ancho;
    this->m = B.m;
}
template <typename T>
void Matriz2D<T>::ReSize(int x, int y){
    if(x != alto || y != ancho){
        this->Limpiar();
        this->Inicializar(x, y);
    }
    this->Zero();
}
template <typename T>
void Matriz2D<T>::Limpiar(){
    if(m == nullptr)
        return;
    for (int i = 0; i < alto; i++) {
        delete[] m[i];
        
    }
    delete[] vectores;
    delete[] m;
    alto = 0;
    ancho = 0;
    m = nullptr;
    vectores = nullptr;
}
template <typename T>
void Matriz2D<T>::Random(){
    int i = 0;
    int i_sig = i + 64;
    if constexpr( std::is_same<T, double>::value == true ) {
        while (i < alto) {
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int k = i; k < i_sig && k < alto; k++) {
                for (int j = 0; j < ancho; j++){
                    std::minstd_rand gen(k+j);
                    std::uniform_real_distribution<double> rango(0.5, 2.5);
                    m[k][j] = rango(gen);
                }
            }
            i = i_sig;
            i_sig += 64;
        }
    }
    else if constexpr( std::is_same<T, float>::value == true ) {
        while (i < alto) {
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int k = i; k < i_sig && k < alto; k++) {
                for (int j = 0; j < ancho; j++){
                    std::minstd_rand gen(k+j);
                    std::uniform_real_distribution<float> rango(0.5f, 2.5f);
                    m[k][j] = rango(gen);
                }
            }
            i = i_sig;
            i_sig += 64;
        }
    }
    else if constexpr( std::is_same<T, int>::value == true ) {
        while (i < alto) {
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int k = i; k < i_sig && k < alto; k++) {
                for (int j = 0; j < ancho; j++){
                    std::minstd_rand gen(k+j);
                    std::uniform_int_distribution<int> rango(1, 100);
                    m[k][j] = rango(gen);
                }
            }
            i = i_sig;
            i_sig += 64;
        }
    }
    else{
        std::cout<<"Aviso: El tipo de dato no es double, no se puede randomizar."<<std::endl;
    }
}
template <typename T>
void Matriz2D<T>::NormalizarFilas(){
    #pragma omp parallel for
    for (int i = 0; i < alto; i++) {
        T promedio = 0.0;
        T varianza = 0.0;
        for (int j = 0; j < ancho; j++) {
            promedio += m[i][j];
        }
        promedio /= ancho;
        for (int j = 0; j < ancho; j++) {
            varianza += (m[i][j] - promedio / ancho) * (m[i][j] - promedio / ancho);
        }
        if (varianza != 0) {
            #pragma omp simd
            for (int j = 0; j < ancho; j++) {
                m[i][j] = (m[i][j] - promedio) / varianza;
            }
        }
    }
}
template <typename T>
void Matriz2D<T>::RELU() {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < alto; i++) {
        for (int j = 0; j < ancho; j++) {
            if (m[i][j] < 0) {
                m[i][j] = 0;
            }
        }
    }
}
template <typename T>
void Matriz2D<T>::SoftmaxFilas() {
    #pragma omp parallel for
    for (int i = 0; i < alto; i++) {
        double valor_maximo = m[i][0];
        for (int j = 1; j < ancho; j++) {
            if (m[i][j] > valor_maximo) {
                valor_maximo = m[i][j];
            }
        }
        double suma_exponentes = 0.0;
        for (int j = 0; j < ancho; j++) {
            m[i][j] = exp(m[i][j] - valor_maximo);
            suma_exponentes += m[i][j];
        }
        #pragma omp simd
        for (int j = 0; j < ancho; j++) {
            m[i][j] /= suma_exponentes;
        }
    }
}
template <typename T>
void Matriz2D<T>::CopiarMatrizDatos(int pos_x, int pos_y, const Matriz2D<T>& B) {
    if (pos_x < 0 || (pos_x + B.alto) > alto || pos_y < 0 || (pos_y + B.ancho) > ancho) {
        std::cerr << "Error: Posicion o dimensiones invalidas." << std::endl;
        return;
    }
    #pragma omp parallel for
    for (int i = 0; i < B.alto; i++){
        for (int j = 0; j < B.ancho; j++){
            m[pos_x + i][pos_y + j] = B.m[i][j];
        }
    }
}
template <typename T>
int Matriz2D<T>::fil(){
    return alto;
}
template <typename T>
int Matriz2D<T>::col(){
    return ancho;
}
template <typename T>
Vector2D<T>& Matriz2D<T>::operator[](int i){
    return vectores[i];
}
template <typename T>
Matriz2D<T>& Matriz2D<T>::operator=(const Matriz2D<T>& B) {
    if (this == &B) 
        return *this;
    if(ancho != B.ancho || alto != B.alto){
        Limpiar();
        this->Inicializar(B.alto, B.ancho);
    }
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < alto; ++i) {
        for (int j = 0; j < ancho; ++j){
            m[i][j] = B.m[i][j];
        }
    }
    return *this;
}
template <typename U>
Matriz2D<U> operator*(const Matriz2D<U>& A, const Matriz2D<U>& B) {
    if (A.ancho != B.alto) {
        std::cerr << "Error: Las matrices no son compatibles para la multiplicación." << std::endl;
        return Matriz2D<U>();
    }
    Matriz2D<U> C(A.alto, B.ancho);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < A.alto; ++i) {
        for (int j = 0; j < B.ancho; ++j) {
            for (int k = 0; k < A.ancho; ++k) {
                C.m[i][j] += A.m[i][k] * B.m[k][j];
            }
        }
    }
    return C;
}
template <typename U>
Matriz2D<U> operator+(const Matriz2D<U>& A, const Matriz2D<U>& B) {
    if (A.alto != B.alto || A.ancho != B.ancho) {
        std::cerr << "Error: Las matrices no son compatibles para la suma." << std::endl;
        return Matriz2D<U>();
    }
    Matriz2D<U> C(A.alto, B.ancho);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < A.alto; ++i) {
        for (int j = 0; j < B.ancho; ++j) {
            C.m[i][j] = A.m[i][j] + B.m[i][j];
        }
    }
    return C;
}
template <typename T>
Matriz2D<T>& Matriz2D<T>::operator*=(const T& escala) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < alto; ++i) {
        for (int j = 0; j < ancho; ++j) {
            m[i][j] *= escala;
        }
    }
    return *this;
}
template <typename T>
Matriz2D<T>& Matriz2D<T>::operator+=(const T& valor) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < alto; ++i) {
        for (int j = 0; j < ancho; ++j) {
            m[i][j] += valor;
        }
    }
    return *this;
}
template <typename T>
Matriz2D<T>& Matriz2D<T>::operator+=(const Vector2D<T>& B) {
    if (ancho != B.largo) {
        std::cerr << "Error: La matriz y el vector no son compatibles para la suma." << std::endl;
        return *this;
    }
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < alto; ++i) {
        for (int j = 0; j < ancho; ++j) {
            m[i][j] += B.v[j];
        }
    }
    return *this;
}
template <typename T>
Matriz2D<T>& Matriz2D<T>::operator+=(const Matriz2D<T>& B) {
    if (alto != B.alto || ancho != B.ancho) {
        std::cerr << "Error: Las matrices no son compatibles para la suma." << std::endl;
        return *this;
    }
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < alto; ++i) {
        for (int j = 0; j < ancho; ++j) {
            m[i][j] += B.m[i][j];
        }
    }
    return *this;
}
template <typename U>
std::ostream& operator<<(std::ostream& os, const Matriz2D<U>& A){
    os<<"["<<A.alto<<" x "<<A.ancho<<"]\n";
    for (int i = 0; i < A.alto; i++){
        os<<'[';
        for (int j = 0; j < A.ancho; j++){
            os<<A.m[i][j];
            if(j != (A.ancho - 1)){
                os<<", ";
            }
        }
        os<<"]\n";
    }
    return os;
}
template <typename T>
Matriz2D<T>::~Matriz2D() {
    for (int i = 0; i < alto; ++i){
        delete[] m[i];
    }
    delete[] m;
}
void imprimir_vector2d(Vector2D<int>& imp){
    std::cout<<"[";
    for (int i = 0; i < imp.lar(); i++){
        std::cout<<imp[i];
        if(i == imp.lar()-1){
            std::cout<<"]\n";
        }else{
            std::cout<<", ";
        }
    }
}
#endif