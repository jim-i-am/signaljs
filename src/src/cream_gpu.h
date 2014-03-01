void cpu_malloc(int** ptr, const int n);

void gpu_malloc(int** ptr, const int n);

void copy_gpu_to_cpu(int** gpu_ptr, int** gpu_ptr, const int n);

void copy_cpu_to_gpu(int** gpu_ptr, int** gpu_ptr, const int n);

void gpu_free(int* ptr);

void cpu_free(int* ptr);

void gpu_fill(int* ptr, int n, int val);

void gpu_seq(int* ptr, int n, int val);

int gpu_sum(int* ptr, int n);

void gpu_sums(int* in_ptr, int* out_ptr,int n);

int gpu_prod(int* ptr, int n);

void gpu_prods(int* in_ptr, int* out_ptr,int n);

int gpu_get(int* ptr, int n, int i);