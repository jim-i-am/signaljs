#include <node.h>
#include "gpuarray.h"

using namespace v8;

void Init(Handle<Object> exports) {
	GpuArray::Init(exports);
}

NODE_MODULE(cream, Init)