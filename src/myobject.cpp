
#include <node.h>
#include <iostream>
#include "gpuarray.h"

using namespace v8;

Persistent<Function> GpuArray::constructor;

GpuArray::GpuArray(double value) : value_(value) {
}

GpuArray::~GpuArray() {
  std::cout << "freed object" << std::endl;
}

void GpuArray::Init(Handle<Object> exports) {
  // Prepare constructor template
  Local<FunctionTemplate> tpl = FunctionTemplate::New(New);
  tpl->SetClassName(String::NewSymbol("GpuArray"));
  tpl->InstanceTemplate()->SetInternalFieldCount(1);
  // Prototype
  tpl->PrototypeTemplate()->Set(String::NewSymbol("copy"),
      FunctionTemplate::New(Copy)->GetFunction());
  constructor = Persistent<Function>::New(tpl->GetFunction());
  exports->Set(String::NewSymbol("GpuArray"), constructor);
}

Handle<Value> GpuArray::New(const Arguments& args) {
  HandleScope scope;

  if (args.IsConstructCall()) {
    // Invoked as constructor: `new GpuArray(...)`
    double value = args[0]->IsUndefined() ? 0 : args[0]->NumberValue();
    GpuArray* obj = new GpuArray(value);
    obj->Wrap(args.This());
    return args.This();
  } else {
    // Invoked as plain function `GpuArray(...)`, turn into construct call.
    const int argc = 1;
    Local<Value> argv[argc] = { args[0] };
    return scope.Close(constructor->NewInstance(argc, argv));
  }
}

Handle<Value> GpuArray::Copy(const Arguments& args) {
  HandleScope scope;

  GpuArray* obj = ObjectWrap::Unwrap<GpuArray>(args.This());
  obj->value_ += 1;

  return scope.Close(New(args));
}