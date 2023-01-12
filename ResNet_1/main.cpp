// Include C++ header files.
#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>
#include <omp.h>
#include <thread>

// Include local CUDA header files.
#include "include/cuda_kernel.cuh"

using std::fstream;
using std::ofstream;
using std::cout;
using std::endl;
using std::to_string;
using std::string;
using std::vector;

#define input_size 5000
#define times 10

int main() {
    int thread_count = 60;
    // vector<vector<float> > conv_in(input_size, vector<float> (3 * 224 * 224, 0.0f));

    float* conv_1_w = new float[64 * 3 * 7 * 7];
    float* conv_1_b = new float[64];

    float* conv_2_w = new float[64 * 64 * 3 * 3];
    float* conv_2_b = new float[64];

    float* conv_3_w = new float[64 * 64 * 3 * 3];
    float* conv_3_b = new float[64];

    float* conv_4_w = new float[64 * 64 * 3 * 3];
    float* conv_4_b = new float[64];

    float* conv_5_w = new float[64 * 64 * 3 * 3];
    float* conv_5_b = new float[64];

    float* conv_6_w = new float[128 * 64 * 3 * 3];
    float* conv_6_b = new float[128];

    float* conv_7_w = new float[128 * 128 * 3 * 3];
    float* conv_7_b = new float[128];

    float* conv_8_w = new float[128 * 64 * 1 * 1];
    float* conv_8_b = new float[128];

    float* conv_9_w = new float[128 * 128 * 3 * 3];
    float* conv_9_b = new float[128];

    float* conv_10_w = new float[128 * 128 * 3 * 3];
    float* conv_10_b = new float[128];

    float* conv_11_w = new float[256 * 128 * 3 * 3];
    float* conv_11_b = new float[256];

    float* conv_12_w = new float[256 * 256 * 3 * 3];
    float* conv_12_b = new float[256];

    float* conv_13_w = new float[256 * 128 * 1 * 1];
    float* conv_13_b = new float[256];

    float* conv_14_w = new float[256 * 256 * 3 * 3];
    float* conv_14_b = new float[256];

    float* conv_15_w = new float[256 * 256 * 3 * 3];
    float* conv_15_b = new float[256];

    float* conv_16_w = new float[512 * 256 * 3 * 3];
    float* conv_16_b = new float[512];

    float* conv_17_w = new float[512 * 512 * 3 * 3];
    float* conv_17_b = new float[512];

    float* conv_18_w = new float[512 * 256 * 1 * 1];
    float* conv_18_b = new float[512];

    float* conv_19_w = new float[512 * 512 * 3 * 3];
    float* conv_19_b = new float[512];

    float* conv_20_w = new float[512 * 512 * 3 * 3];
    float* conv_20_b = new float[512];

    float* fc_w = new float[1000 * 512];
    float* fc_b = new float[1000];

    cout << "parameter start" << endl;

# pragma omp parallel for num_threads(thread_count)
    for (int choice = 1; choice <= 21; choice++) {
        switch (choice)
        {
            case 1 :
                read(conv_1_w, 1, 1, 64, 3, 7, 7);
                read(conv_1_b, 2, 1, 64);
                break;
            case 2 :
                read(conv_2_w, 1, 2, 64, 64, 3, 3);
                read(conv_2_b, 2, 2, 64);
                break;
            case 3 :
                read(conv_3_w, 1, 3, 64, 64, 3, 3);
                read(conv_3_b, 2, 3, 64);
                break;
            case 4 :
                read(conv_4_w, 1, 4, 64, 64, 3, 3);
                read(conv_4_b, 2, 4, 64);
                break;
            case 5 :
                read(conv_5_w, 1, 5, 64, 64, 3, 3);
                read(conv_5_b, 2, 5, 64);
                break;
            case 6 :
                read(conv_6_w, 1, 6, 128, 64, 3, 3);
                read(conv_6_b, 2, 6, 128);
                break;
            case 7 :
                read(conv_7_w, 1, 7, 128, 128, 3, 3);
                read(conv_7_b, 2, 7, 128);
                break;
            case 8 :
                read(conv_8_w, 1, 8, 128, 64, 1, 1);
                read(conv_8_b, 2, 8, 128);
                break;
            case 9 :
                read(conv_9_w, 1, 9, 128, 128, 3, 3);
                read(conv_9_b, 2, 9, 128);
                break;
            case 10 :
                read(conv_10_w, 1, 10, 128, 128, 3, 3);
                read(conv_10_b, 2, 10, 128);
                break;
            case 11 :
                read(conv_11_w, 1, 11, 256, 128, 3, 3);
                read(conv_11_b, 2, 11, 256);
                break;
            case 12 :
                read(conv_12_w, 1, 12, 256, 256, 3, 3);
                read(conv_12_b, 2, 12, 256);
                break;
            case 13 :
                read(conv_13_w, 1, 13, 256, 128, 1, 1);
                read(conv_13_b, 2, 13, 256);
                break;
            case 14 :
                read(conv_14_w, 1, 14, 256, 256, 3, 3);
                read(conv_14_b, 2, 14, 256);
                break;
            case 15 :
                read(conv_15_w, 1, 15, 256, 256, 3, 3);
                read(conv_15_b, 2, 15, 256);
                break;
            case 16 :
                read(conv_16_w, 1, 16, 512, 256, 3, 3);
                read(conv_16_b, 2, 16, 512);
                break;
            case 17 :
                read(conv_17_w, 1, 17, 512, 512, 3, 3);
                read(conv_17_b, 2, 17, 512);
                break;
            case 18 :
                read(conv_18_w, 1, 18, 512, 256, 1, 1);
                read(conv_18_b, 2, 18, 512);
                break;
            case 19 :
                read(conv_19_w, 1, 19, 512, 512, 3, 3);
                read(conv_19_b, 2, 19, 512);
                break;
            case 20 :
                read(conv_20_w, 1, 20, 512, 512, 3, 3);
                read(conv_20_b, 2, 20, 512);
                break;
            case 21 :
                read(fc_w, 3, 20, 1000, 512);
                read(fc_b, 4, 20, 1000);
                break;
            default:
                break;
        }
    }

    cout << "parameter ok" << endl;

    clock_t begin, end;

    // cout << "loading..." << endl;
// # pragma omp parallel for num_threads(thread_count)
//     for (int i = 0; i < input_size; i++) {
//         cout << i << endl;
//         string fname = "./data/input_data/" + to_string(i + 1) + ".txt";
//         fstream infile(fname);

//         for (int x = 0; x < 3; x++) {
//             for (int y = 0; y < 224; y++) {
//                 for (int z = 0; z < 224; z++) {
//                     infile >> conv_in[i][x * 224 * 224 + y * 224 + z];
//                 }
//             }
//         }

        // for (int j = 0; j < times; j++) {
        //     for (int x = 0; x < 3; x++) {
        //         for (int y = 0; y < 224; y++) {
        //             for (int z = 0; z < 224; z++) {
        //                 infile >> conv_in[i * times + j][x * 224 * 224 + y * 224 + z];
        //             }
        //         }
        //     }
        // }
    // }

    // cout << "load ok" << endl;

    for (int i = 4610; i < input_size; i++) {
        cout << "Figure input " << (i + 1) << endl;
        string fname = "./data/input_data/" + to_string(i + 1) + ".txt";
        fstream infile(fname);

        float *conv_in = new float [3 * 224 * 224];

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 224; y++) {
                for (int z = 0; z < 224; z++) {
                    infile >> conv_in[x * 224 * 224 + y * 224 + z];
                }
            }
        }

        begin = clock();

        // Conv 1
        float* conv_1_o = new float[64 * 112 * 112];
        conv(conv_in, conv_1_w, conv_1_b, conv_1_o, 3, 224, 224, 64, 7, 2, 3);

        // ReLU 1
        relu(conv_1_o, 64, 112, 112);
        delete [] conv_in;

        float* maxpool_1_o = new float[64 * 56 * 56];

        maxpool(conv_1_o, maxpool_1_o, 64, 112, 112, 3, 2, 1);
        delete [] conv_1_o;

        // Conv 2
        float* conv_2_o = new float[64 * 56 * 56];
        
        conv(maxpool_1_o, conv_2_w, conv_2_b, conv_2_o, 64, 56, 56, 64, 3, 1, 1);
        relu(conv_2_o, 64, 56, 56);

        // break;
        // Conv 3
        float* conv_3_o = new float[64 * 56 * 56];
        
        conv(conv_2_o, conv_3_w, conv_3_b, conv_3_o, 64, 56, 56, 64, 3, 1, 1);

        delete [] conv_2_o;

        // Add 3
        float* add_1_o = new float[64 * 56 * 56];

        add(maxpool_1_o, conv_3_o, add_1_o, 64, 56, 56);
        delete [] conv_3_o;
        delete [] maxpool_1_o;

        relu(add_1_o, 64, 56, 56);

        // Conv 4
        float* conv_4_o = new float[64 * 56 * 56];
        
        conv(add_1_o, conv_4_w, conv_4_b, conv_4_o, 64, 56, 56, 64, 3, 1, 1);

        relu(conv_4_o, 64, 56, 56);

        // Conv 5
        float* conv_5_o = new float[64 * 56 * 56];
        
        conv(conv_4_o, conv_5_w, conv_5_b, conv_5_o, 64, 56, 56, 64, 3, 1, 1);

        delete [] conv_4_o;

        // Add 2
        float* add_2_o = new float[64 * 56 * 56];

        add(add_1_o, conv_5_o, add_2_o, 64, 56, 56);
        delete [] add_1_o;
        delete [] conv_5_o;

        relu(add_2_o, 64, 56, 56);

        // Conv 6
        float* conv_6_o = new float[128 * 28 * 28];
        
        conv(add_2_o, conv_6_w, conv_6_b, conv_6_o, 64, 56, 56, 128, 3, 2, 1);

        relu(conv_6_o, 128, 28, 28);

        // Conv 7
        float* conv_7_o = new float[128 * 28 * 28];
        
        conv(conv_6_o, conv_7_w, conv_7_b, conv_7_o, 128, 28, 28, 128, 3, 1, 1);

        delete [] conv_6_o;

        // Conv 8
        float* conv_8_o = new float[128 * 28 * 28];
        
        conv(add_2_o, conv_8_w, conv_8_b, conv_8_o, 64, 56, 56, 128, 1, 2, 0);

        delete [] add_2_o;

        // Add 3
        float* add_3_o = new float[128 * 28 * 28];

        add(conv_7_o, conv_8_o, add_3_o, 128, 28, 28);
        delete [] conv_7_o;
        delete [] conv_8_o;

        relu(add_3_o, 128, 28, 28);

        // Conv 9
        float* conv_9_o = new float[128 * 28 * 28];
        
        conv(add_3_o, conv_9_w, conv_9_b, conv_9_o, 128, 28, 28, 128, 3, 1, 1);

        relu(conv_9_o, 128, 28, 28);

        // Conv 10
        float* conv_10_o = new float[128 * 28 * 28];
        
        conv(conv_9_o, conv_10_w, conv_10_b, conv_10_o, 128, 28, 28, 128, 3, 1, 1);

        delete [] conv_9_o;

        // Add 4
        float* add_4_o = new float[128 * 28 * 28];

        add(add_3_o, conv_10_o, add_4_o, 128, 28, 28);
        delete [] add_3_o;
        delete [] conv_10_o;

        relu(add_4_o, 128, 28, 28);

        // Conv 11
        float* conv_11_o = new float[256 * 14 * 14];
        
        conv(add_4_o, conv_11_w, conv_11_b, conv_11_o, 128, 28, 28, 256, 3, 2, 1);

        relu(conv_11_o, 256, 14, 14);

        // Conv 12
        float* conv_12_o = new float[256 * 14 * 14];
        
        conv(conv_11_o, conv_12_w, conv_12_b, conv_12_o, 256, 14, 14, 256, 3, 1, 1);

        delete [] conv_11_o;

        // Conv 13
        float* conv_13_o = new float[128 * 28 * 28];
        
        conv(add_4_o, conv_13_w, conv_13_b, conv_13_o, 128, 28, 28, 256, 1, 2, 0);

        delete [] add_4_o;

        // Add 5
        float* add_5_o = new float[256 * 14 * 14];

        add(conv_12_o, conv_13_o, add_5_o, 256, 14, 14);
        delete [] conv_12_o;
        delete [] conv_13_o;

        relu(add_5_o, 256, 14, 14);

        // Conv 14
        float* conv_14_o = new float[256 * 14 * 14];
        
        conv(add_5_o, conv_14_w, conv_14_b, conv_14_o, 256, 14, 14, 256, 3, 1, 1);

        relu(conv_14_o, 256, 14, 14);

        // Conv 15
        float* conv_15_o = new float[256 * 14 * 14];
        
        conv(conv_14_o, conv_15_w, conv_15_b, conv_15_o, 256, 14, 14, 256, 3, 1, 1);

        delete [] conv_14_o;

        // Add 6
        float* add_6_o = new float[256 * 14 * 14];

        add(add_5_o, conv_15_o, add_6_o, 256, 14, 14);
        delete [] add_5_o;
        delete [] conv_15_o;

        relu(add_6_o, 256, 14, 14);

        // Conv 16
        float* conv_16_o = new float[512 * 7 * 7];
        
        conv(add_6_o, conv_16_w, conv_16_b, conv_16_o, 256, 14, 14, 512, 3, 2, 1);

        relu(conv_16_o, 512, 7, 7);

        // Conv 17
        float* conv_17_o = new float[512 * 7 * 7];
        
        conv(conv_16_o, conv_17_w, conv_17_b, conv_17_o, 512, 7, 7, 512, 3, 1, 1);

        delete [] conv_16_o;

        // Conv 18
        float* conv_18_o = new float[256 * 14 * 14];
        
        conv(add_6_o, conv_18_w, conv_18_b, conv_18_o, 256, 14, 14, 512, 1, 2, 0);

        delete [] add_6_o;

        // Add 7
        float* add_7_o = new float[512 * 7 * 7];

        add(conv_17_o, conv_18_o, add_7_o, 512, 7, 7);
        delete [] conv_17_o;
        delete [] conv_18_o;

        relu(add_7_o, 512, 7, 7);

        // Conv 19
        float* conv_19_o = new float[512 * 7 * 7];
        
        conv(add_7_o, conv_19_w, conv_19_b, conv_19_o, 512, 7, 7, 512, 3, 1, 1);

        relu(conv_19_o, 512, 7, 7);

        // Conv 20
        float* conv_20_o = new float[512 * 7 * 7];
        
        conv(conv_19_o, conv_20_w, conv_20_b, conv_20_o, 512, 7, 7, 512, 3, 1, 1);

        delete [] conv_19_o;

        // Add 8
        float* add_8_o = new float[512 * 7 * 7];

        add(add_7_o, conv_20_o, add_8_o, 512, 7, 7);
        delete [] add_7_o;
        delete [] conv_20_o;

        relu(add_8_o, 512, 7, 7);

        // Global Average Pool
        float* feature = new float[512];

        GlobalAvgPool(add_8_o, feature, 512, 7, 7);

        // Fully Connected Layer
        float* result = new float[1000];
        
        fc(feature, fc_w, fc_b, result, 1000, 512);
        delete [] feature;

        end = clock();

        writetime(double(end - begin) / CLOCKS_PER_SEC, i + 1);

        write2txt(i + 1, result, 1000);
    }

    delete [] conv_1_w;
    delete [] conv_1_b;
    delete [] conv_2_w;
    delete [] conv_2_b;
    delete [] conv_3_w;
    delete [] conv_3_b;
    delete [] conv_4_w;
    delete [] conv_4_b;
    delete [] conv_5_w;
    delete [] conv_5_b;
    delete [] conv_6_w;
    delete [] conv_6_b;
    delete [] conv_7_w;
    delete [] conv_7_b;
    delete [] conv_8_w;
    delete [] conv_8_b;
    delete [] conv_9_w;
    delete [] conv_9_b;
    delete [] conv_10_w;
    delete [] conv_10_b;
    delete [] conv_11_w;
    delete [] conv_11_b;
    delete [] conv_12_w;
    delete [] conv_12_b;
    delete [] conv_13_w;
    delete [] conv_13_b;
    delete [] conv_14_w;
    delete [] conv_14_b;
    delete [] conv_15_w;
    delete [] conv_15_b;
    delete [] conv_16_w;
    delete [] conv_16_b;
    delete [] conv_17_w;
    delete [] conv_17_b;
    delete [] conv_18_w;
    delete [] conv_18_b;
    delete [] conv_19_w;
    delete [] conv_19_b;
    delete [] conv_20_w;
    delete [] conv_20_b;

    return 0;
}