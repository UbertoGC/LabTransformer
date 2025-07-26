#include "Transformer.h"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
Vector2D<int> conversor_img_vec(const Mat& img){
    int divisor = 2;
    int max_r = (img.rows / divisor + 0.5);
    int max_c = (img.cols / divisor + 0.5);
    int n_canal = img.channels();
    Vector2D<int> salida(max_r * max_c * n_canal);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < max_r; i++){
        for (int j = 0; j < max_c; j++){
            Vector2D<int> color(n_canal);
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
                salida[(i * max_c + j)*n_canal + c] = (color[c] / n_pixel + 0.5);
            }
        }
    }
    return salida;
}
int main(){
    int d_modelo = 128;
    int m_entradas = 2352;
    int v_size = 256;
    int m_salidas = 9;
    //{dimension del modelo, maxima entrada, tamaño del vocabulario, numero de cabezas de atencion cruzable}
    int config_transformer[4] = {d_modelo, m_entradas, v_size, m_salidas};
    //{numero de cabezas, dimension del feedforward}
    int config_unificador[2] = {2, 64};
    //{numero de bloques, numero de cabezas, dimension del feedforward}
    int config_bloques[3] = {2, 2, 64};
    int config_decoder[3] = {2, 2, 64};
    int config_encoder[3] = {1, 2, 64};
    string name = "DataTransformer/train_images/train_images_0.png";
    Mat img_prueba = imread(name,1);
    cout<<"ProbandoTiempo"<<endl;
    Vector2D<int> data = conversor_img_vec(img_prueba);
    Transformer<Mat> transformer(conversor_img_vec, config_transformer, config_unificador, config_bloques, config_decoder, config_encoder);
    transformer.Forward(img_prueba);
    return 0;
}