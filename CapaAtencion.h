#ifndef __CAPAATENCION_H__
#define __CAPAATENCION_H__
#include <vector>
#include <string>
#include "Matriz2D.h"
class CapaAtencion{
private:
    Matriz2D<double> atencion_proyeccion;
    Matriz2D<double> atencion_datosconcat;
    Matriz2D<double>* capa_WQ;
    Matriz2D<double>* capa_WK;
    Matriz2D<double>* capa_WV;
    Matriz2D<double>* atencion_matriz;
    Vector2D<double> bias;
    int num_cabezas;
    int t_mascara;
    int d_modelo;
    int d_cabeza;
    void ProyeccionFinal(Matriz2D<double>&);
public:
    CapaAtencion(int, int, int t_m = 0, bool = false);
    void SelfForward(Matriz2D<double>&, Matriz2D<double>&);
    void CrossForward(Matriz2D<double>&, Matriz2D<double>&, Matriz2D<double>&);
    ~CapaAtencion();
};
CapaAtencion::CapaAtencion(int n_c, int d_m, int t_m, bool anunciar) {
    num_cabezas = n_c;
    d_modelo = d_m;
    d_cabeza = d_modelo/num_cabezas;
    t_mascara = t_m;
    capa_WQ = new Matriz2D<double>[num_cabezas];
    capa_WK = new Matriz2D<double>[num_cabezas];
    capa_WV = new Matriz2D<double>[num_cabezas];
    for (int i = 0; i < num_cabezas; i++){
        capa_WQ[i].ReSize(d_modelo, d_cabeza);
        capa_WK[i].ReSize(d_modelo, d_cabeza);
        capa_WV[i].ReSize(d_modelo, d_cabeza);
        capa_WQ[i].Random();
        capa_WK[i].Random();
        capa_WV[i].Random();
    }
    atencion_matriz = nullptr;
    bias.ReSize(d_modelo);
    atencion_proyeccion.ReSize(d_modelo, d_modelo);
    atencion_proyeccion.Random();
    if(anunciar){
        std::cout<<"Capa de Atencion creada"<<std::endl;
    }
}
void CapaAtencion::ProyeccionFinal(Matriz2D<double>& salida){
    salida = atencion_datosconcat * atencion_proyeccion;
    salida += bias;
    if (atencion_matriz != &salida){
        atencion_matriz = &salida;
    }
}
void CapaAtencion::SelfForward(Matriz2D<double>& entrada, Matriz2D<double>& salida){
    atencion_datosconcat.ReSize(entrada.fil(), d_modelo);
    Matriz2D<double> mascara(entrada.fil(), entrada.fil(), t_mascara);
    #pragma omp parallel for
    for (int c = 0; c < num_cabezas; c++) {
        Matriz2D<double> Q = entrada * capa_WQ[c];
        Matriz2D<double> K = entrada * capa_WK[c];
        Matriz2D<double> V = entrada * capa_WV[c];
        Matriz2D<double> puntaje = Q * K.Transpuesta();
        puntaje *= (1.0 / sqrt(d_cabeza));
        puntaje += mascara;
        puntaje.SoftmaxFilas();
        Matriz2D<double> output_cabeza = puntaje * V;
        atencion_datosconcat.CopiarMatrizDatos(0, c * d_cabeza, output_cabeza);
    }
    ProyeccionFinal(salida);
}
void CapaAtencion::CrossForward(Matriz2D<double>& decoder_entrada, Matriz2D<double>& encoder_entrada, Matriz2D<double>& salida){
    atencion_datosconcat.ReSize(decoder_entrada.fil(), d_modelo);
    #pragma omp parallel for
    for (int c = 0; c < num_cabezas; c++) {
        Matriz2D<double> Q = decoder_entrada * capa_WQ[c];
        Matriz2D<double> K = encoder_entrada * capa_WK[c];
        Matriz2D<double> V = encoder_entrada * capa_WV[c];
        Matriz2D<double> puntaje = Q * K.Transpuesta();
        puntaje *= (1.0 / sqrt(d_cabeza));
        puntaje.SoftmaxFilas();
        Matriz2D<double> output_cabeza = puntaje * V;
        atencion_datosconcat.CopiarMatrizDatos(0, c * d_cabeza, output_cabeza);
    }
    ProyeccionFinal(salida);
}
CapaAtencion::~CapaAtencion(){
    delete[] capa_WQ;
    delete[] capa_WK;
    delete[] capa_WV;
}
#endif