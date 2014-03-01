#include <node.h>
#include <v8.h>
#include <iostream>
#include "gpuarray.h"
#include "cream.h"
#include <boost/iostreams/device/mapped_file.hpp>

using namespace v8;

Persistent<Function> GpuArray::constructor;

GpuArray::GpuArray(int n) : n_(n) {
    gpu_malloc(&ptr_, n);
}

GpuArray::~GpuArray() {
    std::cout << "freed object" << std::endl;
    gpu_free(ptr_);
}

void GpuArray::Init(Handle<Object> exports) 
{
    // Prepare constructor template
    Local<FunctionTemplate> tpl = FunctionTemplate::New(New);
    tpl->SetClassName(String::NewSymbol("GpuArray"));
    tpl->InstanceTemplate()->SetInternalFieldCount(1);
    // Prototype
    tpl->PrototypeTemplate()->Set(String::NewSymbol("fill"),
        FunctionTemplate::New(Fill)->GetFunction());
    tpl->PrototypeTemplate()->Set(String::NewSymbol("seq"),
        FunctionTemplate::New(Seq)->GetFunction());
    tpl->PrototypeTemplate()->Set(String::NewSymbol("copy"),
        FunctionTemplate::New(Copy)->GetFunction());  
    tpl->PrototypeTemplate()->Set(String::NewSymbol("sum"),
        FunctionTemplate::New(Sum)->GetFunction());  
    tpl->PrototypeTemplate()->Set(String::NewSymbol("prod"),
        FunctionTemplate::New(Prod)->GetFunction());  
    tpl->PrototypeTemplate()->Set(String::NewSymbol("save"),
        FunctionTemplate::New(Save)->GetFunction());      

    constructor = Persistent<Function>::New(tpl->GetFunction());
    exports->Set(String::NewSymbol("GpuArray"), constructor);
    exports->Set(String::NewSymbol("read"),
        FunctionTemplate::New(Read)->GetFunction());
    exports->Set(String::NewSymbol("save"),
        FunctionTemplate::New(Save)->GetFunction());    
}

Handle<Value> GpuArray::New(const Arguments& args) {
    HandleScope scope;

    if (args.IsConstructCall()) {
        // Invoked as constructor: `new GpuArray(...)`
        int value = args[0]->IsUndefined() ? 0 : args[0]->NumberValue();
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

Handle<Value> GpuArray::Fill(const Arguments& args) {
    HandleScope scope;

    GpuArray* obj = ObjectWrap::Unwrap<GpuArray>(args.This());
    int value = args[0]->IsUndefined() ? 0 : args[0]->NumberValue();
    gpu_fill(obj->ptr_, obj->n_, value);
    
    //Handle<Value> value = GpuArray::New(args);
    //GpuArray* arr = node::ObjectWrap::Unwrap<GpuArray>(value->ToObject());

    return scope.Close(args.This());
}

Handle<Value> GpuArray::Seq(const Arguments& args) {
    HandleScope scope;

    GpuArray* obj = ObjectWrap::Unwrap<GpuArray>(args.This());
    int value = args[0]->IsUndefined() ? 0 : args[0]->NumberValue();
    gpu_seq(obj->ptr_, obj->n_, value);
    
    //Handle<Value> value = GpuArray::New(args);
    //GpuArray* arr = node::ObjectWrap::Unwrap<GpuArray>(value->ToObject());

    return scope.Close(args.This());
}

Handle<Value> GpuArray::Copy(const Arguments& args) {
    HandleScope scope;
    const int argc = 1;

    GpuArray* obj = ObjectWrap::Unwrap<GpuArray>(args.This());
    Local<Value> argv[argc] = { Number::New(obj->n_) };
    return scope.Close(constructor->NewInstance(argc, argv));    
}

Handle<Value> GpuArray::Sum(const Arguments& args) {
    HandleScope scope;
    
    GpuArray* obj = ObjectWrap::Unwrap<GpuArray>(args.This());
    int v = gpu_sum(obj->ptr_, obj->n_);

    return scope.Close(Number::New(v));    
}

Handle<Value> GpuArray::Prod(const Arguments& args) {
    HandleScope scope;
    
    GpuArray* obj = ObjectWrap::Unwrap<GpuArray>(args.This());
    int v = gpu_prod(obj->ptr_, obj->n_);

    return scope.Close(Number::New(v));    
}


Handle<Value> GpuArray::Add(const Arguments& args) {
    HandleScope scope;
    
    /**
    GpuArray* obj = ObjectWrap::Unwrap<GpuArray>(args.This());
    int c = args[0]->IsUndefined() ? 0 : args[0]->NumberValue();
    gpu_add(obj->ptr_, obj->n_, c);
    **/

    return scope.Close(args.This());    
}

Handle<Value> GpuArray::Save(const Arguments& args) {
    HandleScope scope;
    
    GpuArray* obj = ObjectWrap::Unwrap<GpuArray>(args.This());
        // get the param
    v8::String::Utf8Value param1(args[0]->ToString());

    // convert it to string
    std::string path = std::string(*param1); 

    std::cout << "writing to" << path << std::endl;
    const int n = obj->n_;
    boost::iostreams::mapped_file_params params(path);
    params.new_file_size = (n+1) * sizeof(int);

    boost::iostreams::mapped_file_sink sink;
    sink.open(params);
    if (!sink.is_open()) 
    {
        std::cout << "problem opening " << path << std::endl;
        return scope.Close(args.This());
    }

    int* data = (int*) sink.data();
    data[0] = n;
    int* v = 0;
    cpu_malloc(&v, n);
    copy_gpu_to_cpu(obj->ptr_, v, n);
    for (int i = 0; i < n; ++i) {
        data[i+1] = v[i];
    }
    sink.close();

    return scope.Close(args.This());
}

Handle<Value> GpuArray::Read(const Arguments& args) {
    HandleScope scope;

    v8::String::Utf8Value param1(args[0]->ToString());

    // convert it to string
    std::string path = std::string(*param1); 

    std::cout << "reading from " << path << std::endl;

    /**
    std::cout << "file size = " << boost::filesystem::file_size(path) << std::endl;
    boost::iostreams::mapped_file_source src;
    src.open(path);
    if (!src.is_open()) 
    {
        std::cout << "error opening file: " << path <<std::endl;
        return;
    }
    T* data = (int*) src.data();
    const int n = data[0];
    v.resize(n);

    for(int i = 0; i < n; ++i) 
    {
        v[i] = data[i+1];
    }

    src.close();
    **/   
    return scope.Close(args.This());
}