cmd_Release/obj.target/cream.node := flock ./Release/linker.lock g++ -shared -pthread -rdynamic -m64  -Wl,-soname=cream.node -o Release/obj.target/cream.node -Wl,--start-group Release/obj.target/cream/cream.o Release/obj.target/cream/gpuarray.o -Wl,--end-group /home/pez/dev/cream/lib/libcream.so -lboost_system -lboost_filesystem -lboost_iostreams
