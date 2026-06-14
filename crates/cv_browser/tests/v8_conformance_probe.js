function t(name, fn, want){
  var got;
  try { got = fn(); } catch(e){ console.log("ERR  "+name+" => "+e); return; }
  console.log((JSON.stringify(got)===JSON.stringify(want)?"PASS":"FAIL")+" "+name+" => got="+JSON.stringify(got)+" want="+JSON.stringify(want));
}
var f=function(){}; f.a=1; f.b=2;
t("getOwnPropertyNames(fn)", function(){return Object.getOwnPropertyNames(f).filter(function(k){return k==="a"||k==="b";}).sort();}, ["a","b"]);
t("fn.hasOwnProperty", function(){return f.hasOwnProperty("a");}, true);
t("hasOwnProperty.call(fn)", function(){return Object.prototype.hasOwnProperty.call(f,"b");}, true);
t("Reflect.ownKeys(fn) has a", function(){return Reflect.ownKeys(f).indexOf("a")>=0;}, true);
t("Object.create proto", function(){return Object.create({inherited:9},{x:{value:5,enumerable:true}}).inherited;}, 9);
t("key order int-first", function(){var ord={};ord.z=1;ord.a=2;ord[2]=3;ord[1]=4;return Object.keys(ord);}, ["1","2","z","a"]);
t("String.replace fn", function(){return "abc".replace(/b/,function(m){return m.toUpperCase();});}, "aBc");
t("JSON.stringify getter", function(){var go={};Object.defineProperty(go,"v",{enumerable:true,get:function(){return 7;}});return JSON.parse(JSON.stringify(go)).v;}, 7);
t("Object.assign getter", function(){var s={};Object.defineProperty(s,"k",{enumerable:true,get:function(){return 11;}});return Object.assign({},s).k;}, 11);
t("spread getter", function(){var s={};Object.defineProperty(s,"k",{enumerable:true,get:function(){return 13;}});var c={...s};return c.k;}, 13);
t("getOwnPropertyDescriptor accessor", function(){var s={};Object.defineProperty(s,"k",{get:function(){return 1;}});return typeof Object.getOwnPropertyDescriptor(s,"k").get;}, "function");
t("Object.freeze", function(){var x={a:1};Object.freeze(x);x.a=2;return x.a;}, 1);
t("class getter", function(){class C{get x(){return 42;}}return new C().x;}, 42);
t("Map spread entries", function(){return [...new Map([["a",1]])];}, [["a",1]]);
t("getPrototypeOf arr", function(){return Object.getPrototypeOf([])===Array.prototype;}, true);
t("fn.name", function(){return (function named(){}).name;}, "named");
t("bind name", function(){return (function(){}).bind(null).name;}, "bound ");
t("Reflect.get", function(){return Reflect.get({a:5},"a");}, 5);
t("Reflect.has", function(){return Reflect.has({a:1},"a");}, true);
t("Array.from(arraylike)", function(){return Array.from({length:2,0:"a",1:"b"});}, ["a","b"]);
t("String.prototype.matchAll", function(){var r=[];for(var m of "a1b2".matchAll(/\d/g))r.push(m[0]);return r;}, ["1","2"]);
t("Object.defineProperties", function(){var o={};Object.defineProperties(o,{a:{value:1,enumerable:true},b:{value:2,enumerable:true}});return [o.a,o.b];}, [1,2]);
t("setter via defineProperty", function(){var store=0;var o={};Object.defineProperty(o,"x",{set:function(v){store=v*2;},get:function(){return store;}});o.x=5;return o.x;}, 10);
t("computed prop name", function(){var k="dyn";var o={[k]:9};return o.dyn;}, 9);
t("Promise.resolve then", function(){var r="";Promise.resolve(1).then(function(v){r="got"+v;});return r;}, "");
t("typeof Symbol", function(){return typeof Symbol();}, "symbol");
t("Object.values fn-prop-skip", function(){var o={a:1};return Object.values(o);}, [1]);
t("arr destructure default", function(){var [a=5,b=6]=[1];return [a,b];}, [1,6]);
console.log("CONF DONE");
