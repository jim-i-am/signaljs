#ifndef CREAM_H
#define CREAM_H

#include <node.h>

class CreamVector : public node::ObjectWrap {
public:
	int size();
	static void Init(v8::Handle<v8::Object> exports);

private:
	explicit CreamVector(int length = 0);
	CreamVector(CreamVector* v);
	~CreamVector();

	static v8::Handle<v8::Value> New(const v8::Arguments& args);
	static v8::Handle<v8::Value> PlusOne(const v8::Arguments& args);
	static v8::Handle<v8::Value> Fill(const v8::Arguments& args);
	static v8::Handle<v8::Value> Seq(const v8::Arguments& args);
	static v8::Handle<v8::Value> Sum(const v8::Arguments& args);
	static v8::Handle<v8::Value> Sums(const v8::Arguments& args);
	static v8::Handle<v8::Value> Prod(const v8::Arguments& args);
	static v8::Handle<v8::Value> Prods(const v8::Arguments& args);
	static v8::Handle<v8::Value> Get(const v8::Arguments& args);
	static v8::Persistent<v8::Function> constructor;
	int length_;
	int* ptr_;
};

#endif