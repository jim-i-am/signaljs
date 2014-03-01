#include <node.h>

class GpuArray : public node::ObjectWrap {
public:
	static void Init(v8::Handle<v8::Object> exports);
	static v8::Handle<v8::Value> New(const v8::Arguments& args);
	int* ptr_;
	int n_;

private:
	explicit GpuArray(int n = 0);
	~GpuArray();

	static v8::Handle<v8::Value> Fill(const v8::Arguments& args);
	static v8::Handle<v8::Value> Seq(const v8::Arguments& args);
	static v8::Handle<v8::Value> Copy(const v8::Arguments& args);
	static v8::Handle<v8::Value> Sum(const v8::Arguments& args);
	static v8::Handle<v8::Value> Prod(const v8::Arguments& args);
	static v8::Handle<v8::Value> Add(const v8::Arguments& args);  
	static v8::Handle<v8::Value> Save(const v8::Arguments& args);
	static v8::Handle<v8::Value> Read(const v8::Arguments& args); 
	static v8::Persistent<v8::Function> constructor;
};