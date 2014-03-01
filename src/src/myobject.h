#include <node.h>

class GpuArray : public node::ObjectWrap {
 public:
  static void Init(v8::Handle<v8::Object> exports);
  static v8::Handle<v8::Value> New(const v8::Arguments& args);

 private:
  explicit GpuArray(double value = 0);
  ~GpuArray();


  static v8::Handle<v8::Value> Copy(const v8::Arguments& args);
  static v8::Persistent<v8::Function> constructor;
  double value_;
};