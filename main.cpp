#include "Transformer.h"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
Vector2D<double> conversor_img_vec(const Mat& img){
    int divisor = 4;
    int max_r = (img.rows / divisor + 0.5);
    int max_c = (img.cols / divisor + 0.5);
    int n_canal = img.channels();
    Vector2D<double> salida(max_r * max_c * n_canal);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < max_r; i++){
        for (int j = 0; j < max_c; j++){
            Vector2D<double> color(n_canal);
            int n_pixel = 0;
            for (int a = i*divisor; (a < img.rows) && (a < ((i+1)*divisor)); a++){
                for (int b = j*divisor; (b < img.rows) && (b < ((j+1)*divisor)); b++){
                    for (int c = 0; c < n_canal; c++){
                        uchar valor = img.ptr<uchar>(a)[b * img.channels() + c];
                        color[c] += valor;
                    }
                    n_pixel++;
                }
            }
            #pragma omp simd
            for (int c = 0; c < n_canal; c++){
                salida[(i * max_c + j)*n_canal + c] = color[c] / n_pixel;
            }
        }
    }
    return salida;
}
int main(){
    int d_modelo = 128;
    int m_salidas = 9;
    double t_aprendisaje = 5;
    //{dimension del modelo, maxima salida, tasa de aprendisaje}
    int config_transformer[2] = {d_modelo, m_salidas};
    //{numero de cabezas, dimension del feedforward}
    int config_unificador[2] = {4, 64};
    //{numero de bloques, numero de cabezas, dimension del feedforward}
    int config_bloques[3] = {2, 4, 192};
    //{numero de bloques, numero de cabezas, dimension del feedforward, vocabulario size, maxima entrada}
    int config_encoder[5] = {2, 2, 256, 0, 147};
    //{numero de bloques, numero de cabezas, dimension del feedforward, vocabulario size}
    int config_decoder[4] = {2, 2, 96, 0};
    string name = "DataTransformer/train_images/train_images_0.png";
    Mat img_prueba = imread(name,1);
    cout<<"ProbandoTiempo"<<endl;
    Transformer<Mat> transformer(conversor_img_vec, t_aprendisaje, config_transformer, config_unificador, config_bloques, config_decoder, config_encoder);
    transformer.Aprendizaje(img_prueba, 6);
    return 0;
}