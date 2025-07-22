#include "Tensor.h"
#include "Matriz2D.h"
using namespace std;
int main(){
    //double data_a[6] = {-0.165541, 0.117982, -0.884454, 0.187557, 0.0751723, 0.53888};
    //double data_b[6] = {0.490222, 0.956458, -0.686366, 0.0869244, 0.164593, 0.455043};
    //Tensor<double> a(data_a, {2,3});
    //Tensor<double> b(data_b, {3,2});
    Tensor<double> a({800,784}, true);
    Tensor<double> b({784,600}, true);
    cout<<"A---------------------------------------------------------\n";
    imprimir_vector(a.shape());
    //a.imprimir();
    cout<<"B---------------------------------------------------------\n";
    imprimir_vector(b.shape());
    //b.imprimir();
    Tensor<double> c = TensorDot(a,b,{1},{0});
    cout<<"C---------------------------------------------------------\n";
    imprimir_vector(c.shape());
    //c.imprimir();
    Matriz2D<double> alfa(800,784,1);
    cout<<"ALFA------------------------------------------------------\n";
    Matriz2D<double> beta(784,600,1);
    cout<<"BETA------------------------------------------------------\n";
    Matriz2D<double> cade = alfa*beta;
    cout<<"CADE------------------------------------------------------\n";
    
    return 0;
}