#ifndef __MATRIZ2D_H__
#define __MATRIZ2D_H__
#include "Vector2D.h"
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
    Matriz2D(int, int, Matriz2D<T>&);
    Matriz2D(int, int, int = -1);
    Matriz2D(int, int, T**);
    Matriz2D<T> Transpuesta();
    void Transponer();
    void Relacionar(const Matriz2D<T>&);
    void ReSize(int, int);
    void Limpiar();
    void Zero();
    void Random();
    void NormalizarFilasPropias();
    void SoftmaxFilasPropias();
    void RELU();
    void CopiarMatrizDatos(int, int, const Matriz2D<T>&);
    int fil() const;
    int col() const;
    Vector2D<T>& operator[](int) const;
    Matriz2D<T>& operator=(const Matriz2D<T>&);
    Matriz2D<T>& operator*=(const T&);
    Matriz2D<T>& operator/=(const T&);
    Matriz2D<T>& operator+=(const T&);
    Matriz2D<T>& operator+=(const Vector2D<T>&);
    Matriz2D<T>& operator+=(const Matriz2D<T>&);
    Matriz2D<T>& operator-=(const Matriz2D<T>&);
    template <typename U>
    friend std::ostream& operator<<(std::ostream&, const Matriz2D<U>&);
    template <typename U>
    friend Matriz2D<U> operator+(const Matriz2D<U>&, const Matriz2D<U>&);
    template <typename U>
    friend Matriz2D<U> operator*(const Matriz2D<U>&, const Matriz2D<U>&);
    template <typename U>
    friend Matriz2D<U> operator*(const Vector2D<U>&, const Matriz2D<U>&);
    template <typename U>
    friend Matriz2D<U> operator*(const Matriz2D<U>&, const Vector2D<U>&);
    template <typename U>
    friend Matriz2D<U> Matmul(const Matriz2D<U>&, const Matriz2D<U>&);
    template <typename U>
    friend Matriz2D<U> Matmul(const Vector2D<U>&, const Matriz2D<U>&);
    template <typename U>
    friend Matriz2D<U> Matmul(const Matriz2D<U>&, const Vector2D<U>&);
    template <typename U>
    friend Vector2D<U> SumarFilas(const Matriz2D<U>&);
    template <typename U>
    friend Matriz2D<U> SoftmaxFilas(const Matriz2D<U>&);
    template <typename U>
    friend Matriz2D<U> DerSoftmaxFilas(const Matriz2D<U>&, const Matriz2D<U>&);
    template <typename U>
    friend Matriz2D<U> RELU(const Matriz2D<U>&);
    template <typename U>
    friend Matriz2D<U> DerRELU(const Matriz2D<U>&);
    template <typename U>
    friend Matriz2D<U> NormalizarFilas(const Matriz2D<U>&);
    template <typename U>
    friend Matriz2D<U> DerNormalizarFilas(const Matriz2D<U>&, const Matriz2D<U>&);
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
Matriz2D<T>::Matriz2D(int c_inicial, int rango, Matriz2D<T>& B){
    this->Inicializar(B.alto, rango);
    #pragma omp parallel for
    for (int i = 0; i < B.alto; i++){
        #pragma omp simd
        for (int j = 0; j < rango; j++){
            m[i][j] = B.m[i][j + c_inicial];
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
void Matriz2D<T>::NormalizarFilasPropias(){
    #pragma omp parallel for
    for (int i = 0; i < alto; i++) {
        T promedio = 0.0;
        T varianza = 0.0;
        for (int j = 0; j < ancho; j++) {
            promedio += m[i][j];
        }
        promedio /= ancho;
        for (int j = 0; j < ancho; j++) {
            varianza += (m[i][j] - promedio) * (m[i][j] - promedio);
        }
        varianza /= ancho;
        T desviacion_estandar = std::sqrt(varianza);
        if (desviacion_estandar != 0) {
            #pragma omp simd
            for (int j = 0; j < ancho; j++) {
                m[i][j] = (m[i][j] - promedio) / desviacion_estandar;
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
void Matriz2D<T>::SoftmaxFilasPropias() {
    #pragma omp parallel for
    for (int i = 0; i < alto; i++) {
        T valor_maximo = m[i][0];
        for (int j = 1; j < ancho; j++) {
            if (m[i][j] > valor_maximo) {
                valor_maximo = m[i][j];
            }
        }
        T suma_exponentes = 0.0;
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
int Matriz2D<T>::fil()const{
    return alto;
}
template <typename T>
int Matriz2D<T>::col()const{
    return ancho;
}
template <typename T>
Vector2D<T>& Matriz2D<T>::operator[](int i)const{
    return vectores[i];
}
template <typename T>
Matriz2D<T>& Matriz2D<T>::operator=(const Matriz2D<T>& B){
    if (this == &B) 
        return *this;
    if(ancho != B.ancho || alto != B.alto){
        Limpiar();
        this->Inicializar(B.alto, B.ancho);
    }
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < alto; i++) {
        for (int j = 0; j < ancho; ++j){
            m[i][j] = B.m[i][j];
        }
    }
    return *this;
}

template <typename T>
Matriz2D<T>& Matriz2D<T>::operator*=(const T& escala) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < alto; i++) {
        for (int j = 0; j < ancho; ++j) {
            m[i][j] *= escala;
        }
    }
    return *this;
}
template <typename T>
Matriz2D<T>& Matriz2D<T>::operator/=(const T& escala) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < alto; i++) {
        for (int j = 0; j < ancho; ++j) {
            m[i][j] /= escala;
        }
    }
    return *this;
}
template <typename T>
Matriz2D<T>& Matriz2D<T>::operator+=(const T& valor) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < alto; i++) {
        for (int j = 0; j < ancho; ++j) {
            m[i][j] += valor;
        }
    }
    return *this;
}
template <typename T>
Matriz2D<T>& Matriz2D<T>::operator+=(const Vector2D<T>& B) {
    if (ancho != B.largo) {
        std::cerr << "Error: La matriz y el vector no son compatibles para la suma: " << this->alto << " X " << this->ancho<< " - " << B.lar() << " X " << 1 << ".\n";
        return *this;
    }
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < alto; i++) {
        for (int j = 0; j < ancho; ++j) {
            m[i][j] += B.v[j];
        }
    }
    return *this;
}
template <typename T>
Matriz2D<T>& Matriz2D<T>::operator+=(const Matriz2D<T>& B) {
    if (alto != B.alto || ancho != B.ancho) {
        std::cerr << "Error: Las matrices no son compatibles para la suma: " << this->alto << " X " << this->ancho<< " - " << B.alto << " X " << B.ancho << ".\n";
        return *this;
    }
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < alto; i++) {
        for (int j = 0; j < ancho; ++j) {
            m[i][j] += B.m[i][j];
        }
    }
    return *this;
}
template <typename T>
Matriz2D<T>& Matriz2D<T>::operator-=(const Matriz2D<T>& B) {
    if (alto != B.alto || ancho != B.ancho) {
        std::cerr << "Error: Las matrices no son compatibles para la resta: "<< this->alto << " X " << this->ancho<< " - " << B.alto << " X " << B.ancho << ".\n";
        return *this;
    }
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < alto; i++) {
        for (int j = 0; j < ancho; ++j) {
            m[i][j] -= B.m[i][j];
        }
    }
    return *this;
}
template <typename U>
std::ostream& operator<<(std::ostream& os, const Matriz2D<U>& A){
    os<<"["<<A.alto<<" x "<<A.ancho<<"]\n";
    if(std::max(A.alto, A.ancho) < 5){
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
    }
    return os;
}
template <typename U>
Matriz2D<U> operator+(const Matriz2D<U>& A, const Matriz2D<U>& B) {
    if (A.alto != B.alto || A.ancho != B.ancho) {
        std::cerr << "Error: Las matrices no son compatibles para la suma.";
        std::cout<<A<<B;
        return Matriz2D<U>();
    }
    Matriz2D<U> C(A.alto, B.ancho);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < A.alto; i++) {
        for (int j = 0; j < B.ancho; j++) {
            C.m[i][j] = A.m[i][j] + B.m[i][j];
        }
    }
    return C;
}
template <typename U>
Matriz2D<U> operator*(const Matriz2D<U>& A, const Matriz2D<U>& B) {
    if (A.alto != B.alto || A.ancho != B.ancho) {
        std::cerr << "Error: Las matrices no son compatibles para la multiplicacion." << std::endl;
        return Matriz2D<U>();
    }
    Matriz2D<U> C(A.alto, B.ancho);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < A.alto; i++) {
        for (int j = 0; j < B.ancho; j++) {
            C.m[i][j] = A.m[i][j] * B.m[i][j];
        }
    }
    return C;
}
template <typename U>
Matriz2D<U> operator*(const Vector2D<U>& A, const Matriz2D<U>& B) {
    if (A.lar() != B.alto) {
        std::cerr << "Error: Las matrices no son compatibles para la multiplicación." << std::endl;
        return Matriz2D<U>();
    }
    Matriz2D<U> C(B.alto, B.ancho);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < B.alto; i++){
        for (int j = 0; j < B.ancho; j++){
            C.m[i][j] = B.m[i][j] * A[i];
        }
    }
    return C;
}
template <typename U>
Matriz2D<U> operator*(const Matriz2D<U>& A, const Vector2D<U>& B) {
    if (B.lar() != A.ancho) {
        std::cerr << "Error: Las matrices no son compatibles para la multiplicación." << std::endl;
        return Matriz2D<U>();
    }
    Matriz2D<U> C(A.alto, A.ancho);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < A.alto; i++){
        for (int j = 0; j < A.ancho; j++){
            C.m[i][j] = B[j] * A.m[i][j];
        }
    }
    return C;
}

template <typename U>
Matriz2D<U> Matmul(const Matriz2D<U>& A, const Matriz2D<U>& B) {
    if (A.ancho != B.alto) {
        std::cout << "Error: Las matrices no son compatibles para Matmul." << std::endl;
        return Matriz2D<U>();
    }
    Matriz2D<U> C(A.alto, B.ancho);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < A.alto; i++) {
        for (int j = 0; j < B.ancho; ++j) {
            for (int k = 0; k < A.ancho; ++k) {
                C.m[i][j] += A.m[i][k] * B.m[k][j];
            }
        }
    }
    return C;
}
template <typename U>
Matriz2D<U> Matmul(const Vector2D<U>& A, const Matriz2D<U>& B) {
    if (A.lar() != B.alto) {
        std::cerr << "Error: Las matrices no son compatibles para la Matmul." << std::endl;
        return Matriz2D<U>();
    }
    Matriz2D<U> C(1, B.ancho);
    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < B.ancho; ++j) {
        for (int k = 0; k < A.lar(); ++k) {
            C.m[0][j] += A[k] * B.m[k][j];
        }
    }
    return C;
}
template <typename U>
Matriz2D<U> Matmul(const Matriz2D<U>& A, const Vector2D<U>& B) {
    if (A.ancho != 1) {
        std::cerr << "Error: Las matrices no son compatibles para la Matmul." << std::endl;
        return Matriz2D<U>();
    }
    Matriz2D<U> C(A.alto, B.lar());
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < A.alto; i++) {
        for (int j = 0; j < B.lar(); ++j) {
            C.m[i][j] += A.m[i][0] * B[j];
        }
    }
    return C;
}
template <typename U>
Vector2D<U> SumarFilas(const Matriz2D<U>& A){
    Vector2D<U> C(A.alto);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < A.alto; i++) {
        for (int j = 0; j < A.ancho; j++) {
            C[i] += A[i][j];
        }
    }
    return C;
}
template <typename U>
Matriz2D<U> SoftmaxFilas(const Matriz2D<U>& A){
    Matriz2D<U> C(A.alto, A.ancho);
    #pragma omp parallel for
    for (int i = 0; i < A.alto; i++) {
        double valor_maximo = A.m[i][0];
        for (int j = 1; j < A.ancho; j++) {
            if (A.m[i][j] > valor_maximo) {
                valor_maximo = A.m[i][j];
            }
        }
        double suma_exponentes = 0.0;
        for (int j = 0; j < A.ancho; j++) {
            C.m[i][j] = exp(A.m[i][j] - valor_maximo);
            suma_exponentes += C.m[i][j];
        }
        #pragma omp simd
        for (int j = 0; j < A.ancho; j++) {
            C.m[i][j] /= suma_exponentes;
        }
    }
    return C;
}
template <typename U>
Matriz2D<U> DerSoftmaxFilas(const Matriz2D<U>& A,  const Matriz2D<U>& grad_sig){
    Matriz2D<U> C(A.alto, A.ancho);
    #pragma omp parallel for
    for (int i = 0; i < A.alto; i++) {
        U matmul_resultado = 0;
        for (int j = 0; j < A.ancho; j++) {
            matmul_resultado += grad_sig.m[i][j] * A.m[i][j];
        }
        for (int j = 0; j < A.ancho; j++) {
            C.m[i][j] = A.m[i][j] * (grad_sig.m[i][j] - matmul_resultado);
        }
    }
    return C;
}
template <typename U>
Matriz2D<U> RELU(const Matriz2D<U>& A){
    Matriz2D<U> C(A.alto, A.ancho);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < A.alto; i++) {
        for (int j = 0; j < A.ancho; j++) {
            if (A.m[i][j] <= 0) {
                C.m[i][j] = 0;
            }else{
                C.m[i][j] = A.m[i][j];
            }
        }
    }
    return C;
}
template <typename U>
Matriz2D<U> DerRELU(const Matriz2D<U>& A){
    Matriz2D<U> C(A.alto, A.ancho);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < A.alto; i++) {
        for (int j = 0; j < A.ancho; j++) {
            if (A.m[i][j] <= 0) {
                C.m[i][j] = 0;
            }else{
                C.m[i][j] = 1;
            }
        }
    }
    return C;
}
template <typename U>
Matriz2D<U> NormalizarFilas(const Matriz2D<U>& A){
    Matriz2D<U> C(A.alto, A.ancho);
    #pragma omp parallel for
    for (int i = 0; i < A.alto; i++) {
        U promedio = 0;
        U varianza = 0;
        for (int j = 0; j < A.ancho; j++) {
            promedio += A.m[i][j];
        }
        promedio /= A.ancho;
        for (int j = 0; j < A.ancho; j++) {
            varianza += (A.m[i][j] - promedio) * (A.m[i][j] - promedio);
        }
        varianza /= A.ancho;
        U desviacion_estandar = std::sqrt(varianza);
        if (varianza > 0 && desviacion_estandar > 0) {
            #pragma omp simd
            for (int j = 0; j < A.ancho; j++) {
                C.m[i][j] = (A.m[i][j] - promedio) / desviacion_estandar;
            }
        }
    }
    return C;
}
template <typename U>
Matriz2D<U> DerNormalizarFilas(const Matriz2D<U>& A, const Matriz2D<U>& grad_sig){
    Matriz2D<U> C(A.alto, A.ancho);
    #pragma omp parallel for
    for (int i = 0; i < A.alto; i++) {
        U promedio = 0;
        U varianza = 0;
        for (int j = 0; j < A.ancho; j++) {
            promedio += A.m[i][j];
        }
        promedio /= A.ancho;
        for (int j = 0; j < A.ancho; j++) {
            varianza += (A.m[i][j] - promedio) * (A.m[i][j] - promedio);
        }
        varianza /= A.ancho;
        U desviacion_estandar = std::sqrt(varianza);
        if (varianza > 0 && desviacion_estandar > 0) {
            Vector2D<U> x_hat(A.ancho);
            for (int j = 0; j < A.ancho; j++){
                x_hat[j] = (A.m[i][j] - promedio) / desviacion_estandar;
            }
            U sum_dy = 0.0;
            U sum_dy_xhat = 0.0;
            for (int j = 0; j < A.ancho; j++) {
                sum_dy += grad_sig.m[i][j];
                sum_dy_xhat += grad_sig.m[i][j] * x_hat[j];
            }
            #pragma omp simd
            for (int j = 0; j < A.ancho; j++) {
                C.m[i][j] = (grad_sig.m[i][j] * A.ancho - sum_dy - x_hat[j] * sum_dy_xhat) / (A.ancho * desviacion_estandar);
            }
        }
    }
    return C;
}
template <typename T>
Matriz2D<T>::~Matriz2D() {
    for (int i = 0; i < alto; i++){
        delete[] m[i];
    }
    delete[] m;
}
void imprimir_vector2d(Vector2D<double>& imp){
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
#endif