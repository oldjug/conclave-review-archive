console.log(
  "PROMISE_PROTO_PROBE",
  JSON.stringify({
    typeofPromise: typeof Promise,
    typeofPromiseProto: typeof Promise.prototype,
    protoKeys: Promise.prototype ? Object.getOwnPropertyNames(Promise.prototype) : [],
    typeofFinally: Promise.prototype ? typeof Promise.prototype.finally : "missing-proto",
    typeofFinallyCall:
      Promise.prototype && Promise.prototype.finally
        ? typeof Promise.prototype.finally.call
        : "missing-finally",
  })
);
