// Conclave bench — integer loop microbench (body of bench_loop.html, de-throwed).
// Fixed nested integer loop. Returns the sum via assignment to a global so
// run() completes cleanly (no throw). Known T2-JIT exerciser.
var fb = (function () {}) instanceof Object;
function work(n) {
  var s = 0;
  for (var i = 0; i < n; i = i + 1) {
    s = s + i;
  }
  return s;
}
var __bench_loop_result = 0;
for (var j = 0; j < 6000; j++) {
  __bench_loop_result = __bench_loop_result + work(400);
}
