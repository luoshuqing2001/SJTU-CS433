// cpp functions called in the main.cpp
void write2txt(int i, float* p, int a);

void writetime(double t, int i);

void read(float* array, int mode, int serial, int a, int b = 0, int c = 0, int d = 0);

void conv(float *input_raw, float *w, float *b, float *output, 
          int i_1, int i_2, int i_3, int chl, int krl, int stride, int pad);

void relu(float* input, int a, int b, int c);

void maxpool(float* input_raw, float* output, int a, int b, int c, int krl, int stride, int pad);

void add(const float* A, const float* B, float* C, int a, int b, int c);

void GlobalAvgPool(float* input, float* output, int a, int b, int c);

void fc(float* feature, float* fc_w, float* fc_b, float* result, int na, int nb);

