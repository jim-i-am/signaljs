class GpuVector {
public:
	GpuVector(int n);
	~GpuVector();
	int size();

private:
	int* ptr;
	int length;
};