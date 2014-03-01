#include <node.h>
#include "cream_vector.hpp"
#include "cream_cuda.h"

using namespace v8;

Persistent<Function> CreamVector::constructor;

CreamVector::CreamVector(int length) : length_(length) {
	gpu_malloc(&ptr_, length);
	V8::AdjustAmountOfExternalAllocatedMemory(size());
}

CreamVector::CreamVector(CreamVector* vec) {
	CreamVector(vec->length_);
}

CreamVector::~CreamVector() {
	V8::AdjustAmountOfExternalAllocatedMemory(-size());
	gpu_free(ptr_);
}

int CreamVector::size() {
	return length_ * sizeof(int);
}

void CreamVector::Init(Handle<Object> exports) {
  // Prepare constructor template
  Local<FunctionTemplate> tpl = FunctionTemplate::New(New);
  
  tpl->SetClassName(String::NewSymbol("CreamVector"));
  
  tpl->InstanceTemplate()->SetInternalFieldCount(1);
  // Prototype
  
  tpl->PrototypeTemplate()->Set(String::NewSymbol("plusOne"),
      FunctionTemplate::New(PlusOne)->GetFunction());
  
  tpl->PrototypeTemplate()->Set(String::NewSymbol("fill"), 
  	FunctionTemplate::New(Fill)->GetFunction());

  tpl->PrototypeTemplate()->Set(String::NewSymbol("sum"), 
  	FunctionTemplate::New(Sum)->GetFunction());  

  tpl->PrototypeTemplate()->Set(String::NewSymbol("sums"), 
  	FunctionTemplate::New(Sums)->GetFunction());    

  tpl->PrototypeTemplate()->Set(String::NewSymbol("prod"), 
  	FunctionTemplate::New(Prod)->GetFunction());  

  tpl->PrototypeTemplate()->Set(String::NewSymbol("prods"), 
  	FunctionTemplate::New(Prods)->GetFunction());    
  
    tpl->PrototypeTemplate()->Set(String::NewSymbol("seq"), 
      FunctionTemplate::New(Seq)->GetFunction());

  tpl->PrototypeTemplate()->Set(String::NewSymbol("get"), 
      FunctionTemplate::New(Get)->GetFunction()); 

  constructor = Persistent<Function>::New(tpl->GetFunction());
  exports->Set(String::NewSymbol("CreamVector"), constructor);
}

Handle<Value> CreamVector::New(const Arguments& args) {
  HandleScope scope;

  if (args.IsConstructCall()) {
    // Invoked as constructor: `new CreamVector(...)`
    int value = args[0]->IsUndefined() ? 0 : args[0]->NumberValue();
    CreamVector* obj = new CreamVector(value);
    obj->Wrap(args.This());
    return args.This();
  } else {
    // Invoked as plain function `CreamVector(...)`, turn into construct call.
    const int argc = 1;
    Local<Value> argv[argc] = { args[0] };
    return scope.Close(constructor->NewInstance(argc, argv));
  }
}

Handle<Value> CreamVector::PlusOne(const Arguments& args) {
  HandleScope scope;

  CreamVector* obj = ObjectWrap::Unwrap<CreamVector>(args.This());
  obj->length_ += 1;

  return scope.Close(Number::New(obj->length_));
}

Handle<Value> CreamVector::Fill(const Arguments& args) {
  HandleScope scope;

  if (args.Length() != 1) {
  	ThrowException(Exception::TypeError(String::New("Wrong number of arguments")));
  }

  if (!args[0]->IsNumber()) {
  	ThrowException(Exception::TypeError(String::New("Wrong argument")));
    return scope.Close(Undefined());
  }

  CreamVector* obj = ObjectWrap::Unwrap<CreamVector>(args.This());
  int value = args[0]->IsUndefined() ? 0 : args[0]->NumberValue();
  gpu_fill(obj->ptr_, obj->length_, value);

  return scope.Close(Undefined());
}

Handle<Value> CreamVector::Seq(const Arguments& args) {
  HandleScope scope;

  if (args.Length() != 1) {
    ThrowException(Exception::TypeError(String::New("Wrong number of arguments")));
  }

  if (!args[0]->IsNumber()) {
    ThrowException(Exception::TypeError(String::New("Wrong argument")));
    return scope.Close(Undefined());
  }

  CreamVector* obj = ObjectWrap::Unwrap<CreamVector>(args.This());
  int value = args[0]->IsUndefined() ? 0 : args[0]->NumberValue();
  gpu_seq(obj->ptr_, obj->length_, value);

  return scope.Close(Undefined());
}

Handle<Value> CreamVector::Sum(const Arguments& args) {
  HandleScope scope;

  CreamVector* obj = ObjectWrap::Unwrap<CreamVector>(args.This());
  int sum = gpu_sum(obj->ptr_, obj->length_);

  return scope.Close(Number::New(sum));
}

Handle<Value> CreamVector::Sums(const Arguments& args) {
	HandleScope scope;

	CreamVector* in = ObjectWrap::Unwrap<CreamVector>(args.This());
	CreamVector* out = new CreamVector(in->length_); 
	
  gpu_sums(in->ptr_, out->ptr_, out->length_);

  out->Wrap(args.This());
  return args.This();
}

Handle<Value> CreamVector::Prod(const Arguments& args) {
  HandleScope scope;

  CreamVector* obj = ObjectWrap::Unwrap<CreamVector>(args.This());
  int prod = gpu_prod(obj->ptr_, obj->length_);

  return scope.Close(Number::New(prod));
}

Handle<Value> CreamVector::Prods(const Arguments& args) {
	HandleScope scope;

	CreamVector* in = ObjectWrap::Unwrap<CreamVector>(args.This());
	CreamVector* out = new CreamVector(in->length_); 
	gpu_prods(in->ptr_, out->ptr_, in->length_);

  out->Wrap(args.This());
  args.This()->SetInternalField(0, out);

  return args.This();
 /** HandleScope scope;
     
    Local<Object> obj = Object::New();
    obj->Set(String::NewSymbol("x"), Number::New( 1 ));
    obj->Set(String::NewSymbol("y"), Number::New( 1 ));
 
    return scope.Close(obj);**/
    /**HandleScope scope;
    Handle<ObjectTemplate> point_templ = ObjectTemplate::New();
    CreamVector* v = new CreamVector(1000000);
    point_templ->SetInternalFieldCount(1);
    Local<Object> obj = point_templ->NewInstance();
    obj->SetInternalField(0, External::New(v));
    return scope.Close(obj);**/
}

Handle<Value> CreamVector::Get(const Arguments& args) {
  HandleScope scope;

  if (args.Length() != 1) {
    ThrowException(Exception::TypeError(String::New("Wrong number of arguments")));
  }

  if (!args[0]->IsNumber()) {
    ThrowException(Exception::TypeError(String::New("Wrong argument")));
    return scope.Close(Undefined());
  }

  CreamVector* obj = ObjectWrap::Unwrap<CreamVector>(args.This());
  int i = args[0]->IsUndefined() ? 0 : args[0]->NumberValue();
  int val = gpu_get(obj->ptr_, obj->length_, i);

  return scope.Close(Number::New(val));
}