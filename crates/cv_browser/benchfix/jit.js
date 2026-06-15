// Conclave bench — float-heavy loop microbench (body of bench_jit.html, de-throwed).
// 1.5M-iter float accumulation through an arithmetic-heavy function. Returns the
// sum via assignment to a global so run() completes cleanly (no throw). Known
// T2-JIT exerciser (unboxed f64 fast path).
var fb = (function () {}) instanceof Object;
function f(x) {
  return ((x * x * 0.5 + x * 3.0 - 1.0) * (x - 2.0) + x * x * x * 0.25) / (x + 1.0) - x * 0.5 + x * x * 0.125 - x * 7.0;
}
var __bench_jit_result = 0;
for (var i = 0; i < 1500000; i++) {
  __bench_jit_result = __bench_jit_result + f(i);
}
