#include "Transformer.h"
#include <opencv2/opencv.hpp>
#include <fstream>
using namespace std;
using namespace cv;
Vector2D<int> conversor_img_vec(const Mat& img){
    int divisor = 7;
    int max_r = (img.rows / divisor + 0.5);
    int max_c = (img.cols / divisor + 0.5);
    int n_canal = img.channels();
    Vector2D<int> salida(max_r * max_c * n_canal);
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
                salida[(i * max_c + j)*n_canal + c] = int((color[c] / n_pixel) + 0.5);
            }
        }
    }
    return salida;
}
void leer_img(string dir, vector<Mat>& imgs){
    for (int i = 0; i < imgs.size(); i++){
        string dir_final = dir + to_string(i) + ".png";
        imgs[i] = imread(dir_final,1);
    }
}
void leer_labels(string dir, vector<int>& clases){
    int id = 0;
    ifstream archivo(dir);
    string linea;
    while (getline(archivo, linea) && id < clases.size()) {
        clases[id] = stoi(linea);
        id++;
    }
    archivo.close();
}
void Probar(Transformer<Mat, float>& transformer){
    int acertados = 0;
    int fallados = 0;
    int limit_val = 1000;
    vector<int> clase_val(limit_val);
    vector<Mat> img_val(limit_val);
    string val_lab_dir = "DataTransformer/val_labels/val_labels.txt";
    string val_imgdir = "DataTransformer/val_images/val_images_";
    leer_labels(val_lab_dir, clase_val);
    leer_img(val_imgdir, img_val);
    cout<<"Ejecutando ...\n";
    for (int i = 0; i < limit_val; i++){
        int r = transformer.Ejecutar(img_val[i]);
        if(r == clase_val[i]){
            acertados++;
        }else{
            fallados++;
        }   
    }
    clase_val.clear();
    img_val.clear();
    cout<<"Acertados: "<<acertados<<", Fallados: "<<fallados<<endl;
    cout<<"Porcentaje de certeza: "<<double(acertados) / double(limit_val)<<endl;
}
int main(){
    int d_modelo = 32;
    int m_salidas = 8;
    float t_aprendisaje = 0.005;
    //{dimension del modelo, maxima salida, tasa de aprendisaje}
    int config_transformer[2] = {d_modelo, m_salidas};
    //{numero de cabezas, dimension del feedforward}
    int config_unificador[2] = {2, 64};
    //{numero de bloques, numero de cabezas, dimension del feedforward}
    int config_bloques[3] = {1, 2, 48};
    //{numero de bloques, numero de cabezas, dimension del feedforward, vocabulario size, maxima entrada}
    int config_encoder[5] = {1, 2, 48, 0, 48};
    //{numero de bloques, numero de cabezas, dimension del feedforward, vocabulario size}
    
    int config_decoder[4] = {1, 2, 64, 0};
    
    Transformer<Mat, float> transformer(conversor_img_vec, t_aprendisaje, config_transformer, config_unificador, config_bloques, config_decoder, config_encoder);
    //Maximo 11959
    int epocas = 30;
    int limit_train = 100;
    int clock = limit_train/10;
    vector<int> clase_train(limit_train);
    vector<Mat> img_train(limit_train);
    string train_lab_dir = "DataTransformer/train_labels/train_labels.txt";
    string train_imgdir = "DataTransformer/train_images/train_images_";
    leer_labels(train_lab_dir, clase_train);
    leer_img(train_imgdir, img_train);
    cout<<"Aprendiendo ...\n";
    for (int e = 0; e < epocas; e++){
        cout<<"Epoca: "<<e<<" ";
        for (int i = 0; i < limit_train; i++){
            transformer.Aprendizaje(img_train[i], clase_train[i]);
            if(i % clock == 0){
                cout<<"#";
            }
        }
        cout<<endl;
    }
    Probar(transformer);
    clase_train.clear();
    img_train.clear();

    
    return 0;
}