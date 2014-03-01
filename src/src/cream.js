var cream = require('./build/Release/cream');

var n = 10000000;
var v = cream.GpuArray(n);
cream.read("read.crm");
v.fill(1);
v.save("test.crm");
/**
for (var i = 0; i < n; i++) {
	console.log(v.sum());
	console.log(v.prod());
}
**/

/**
var cream = require('./build/Release/cream');

var n = 1000000;

var r = new cream.CreamVector(n);
for (var i = 0; i < n; ++i) {
	var v = r.prods();
}


var v = new cream.CreamVector(n);
v.seq(0);

console.log('sum = '+ v.sum());

var s2 = v.sums().sum();
console.log('scan = '+ s2);

v.fill(1);
console.log('prod = '+ v.prod());

var p = v.prods();
console.log('prods = '+ v.prods().sum());

var s = v.sums();
console.log('sums = '+ v.sums().sum());

v = new cream.CreamVector(n);
v.seq(0);

console.log("0="+v.get(0));
console.log("n/2="+v.get(n/2));
console.log("n-1="+v.get(n-1));
**/