//! A faithful MINIMAL `testharness.js` shim for running Web Platform Tests in
//! the engine without a real WPT checkout.
//!
//! HONESTY NOTE: this is **not** the vendored upstream `testharness.js`. It is a
//! hand-written re-implementation of the subset of the WPT testharness API that
//! the runnable test classes (DOM, CSS computed-value, HTML parsing) actually
//! use: `test()`, `async_test()`, `promise_test()`, the `assert_*` family, the
//! `Test` step helpers, `setup()`, `done()`, and the completion-callback plumbing.
//! It is implemented to match WPT semantics (each `test()` is an independent
//! pass/fail unit; an `assert_*` throws an `AssertionError`/`OptionalFeatureUnsupportedError`
//! that the harness catches and records against the current test).
//!
//! It deliberately does NOT implement: the visual `<div id=log>` reporter,
//! `RemoteContext`, `fetch_tests_from_worker`, the IDL harness (`idlharness.js`),
//! `testdriver`/`test_driver` automation, or reftests. Tests that need those are
//! reported honestly (their assertions throw `ReferenceError` and the test fails,
//! or the runner SKIPs them when it can detect the dependency up front).
//!
//! Results are accumulated on `globalThis.__wpt`. After all scripts run and the
//! scheduler drains, the Rust side reads `__wpt.serialize()` to harvest the
//! per-subtest pass/fail records and the harness status.

/// The JS source injected as the FIRST inline script of every WPT test document,
/// before any of the test's own scripts run.
pub(crate) const TESTHARNESS_SHIM: &str = r#"
(function(){
  'use strict';
  if (globalThis.__wpt) { return; } // idempotent

  // ---- harness state ------------------------------------------------------
  var STATUS = { OK: 0, ERROR: 1, TIMEOUT: 2, PRECONDITION_FAILED: 3 };
  var TEST_STATUS = { PASS: 0, FAIL: 1, TIMEOUT: 2, NOTRUN: 3, PRECONDITION_FAILED: 4 };

  var tests = [];                 // all Test objects, in creation order
  var completion_callbacks = [];
  var result_callbacks = [];
  var harness_status = { status: STATUS.OK, message: null };
  var all_loaded = false;         // window load fired
  var phase_complete = false;     // done() / auto-complete reached
  var single_test_mode = false;   // generate_tests / test()-less docs
  var settings = { explicit_done: false, explicit_timeout: false, single_test: false };

  // ---- error types --------------------------------------------------------
  function AssertionError(message){ this.message = message; this.stack = (new Error()).stack; }
  AssertionError.prototype = Object.create(Error.prototype);
  AssertionError.prototype.name = 'AssertionError';
  AssertionError.prototype.toString = function(){ return 'AssertionError: ' + this.message; };

  function OptionalFeatureUnsupportedError(message){ this.message = message; this.stack = (new Error()).stack; }
  OptionalFeatureUnsupportedError.prototype = Object.create(AssertionError.prototype);
  OptionalFeatureUnsupportedError.prototype.name = 'OptionalFeatureUnsupportedError';

  // ---- value formatting (for assertion messages) --------------------------
  function format_value(val){
    try {
      if (val === null) return 'null';
      if (val === undefined) return 'undefined';
      var t = typeof val;
      if (t === 'string') return '"' + val.replace(/\\/g,'\\\\').replace(/"/g,'\\"') + '"';
      if (t === 'number') {
        if (val === 0 && 1/val === -Infinity) return '-0';
        return String(val);
      }
      if (t === 'boolean' || t === 'bigint' || t === 'symbol') return String(val);
      if (t === 'function') return 'function "' + (val.name || '') + '"';
      if (Array.isArray(val)) {
        var parts = [];
        for (var i=0;i<val.length && i<10;i++) parts.push(format_value(val[i]));
        if (val.length > 10) parts.push('...');
        return '[' + parts.join(', ') + ']';
      }
      if (t === 'object') {
        if (val.nodeType !== undefined && val.nodeName !== undefined) {
          return 'Element node <' + String(val.nodeName).toLowerCase() + '>';
        }
        var ctor = (val.constructor && val.constructor.name) || 'Object';
        return 'object "[object ' + ctor + ']"';
      }
      return String(val);
    } catch(e){ return '[value]'; }
  }

  function make_message(function_name, description, error, substitutions){
    var msg = function_name + ':';
    if (description) msg += ' ' + description;
    if (error) {
      var e = error;
      if (substitutions) {
        for (var k in substitutions){
          e = e.split('${' + k + '}').join(format_value(substitutions[k]));
        }
      }
      msg += ' ' + e;
    }
    return msg;
  }

  function same_value(x, y){
    // SameValueZero-ish but treat NaN==NaN true and +0/-0 equal (assert_equals
    // uses ===, but we special-case NaN here to match testharness).
    if (x === y){ return true; }
    if (x !== x && y !== y){ return true; } // NaN
    return false;
  }

  // ---- the Test object ----------------------------------------------------
  function Test(name, properties){
    this.name = name;
    this.status = TEST_STATUS.NOTRUN;
    this.message = null;
    this.stack = null;
    this.is_async = false;
    this.cleanup_callbacks = [];
    this.phase_completed = false;
    this.properties = properties || {};
    tests.push(this);
  }
  Test.prototype.structured_clone = function(){
    return { name: this.name, status: this.status, message: this.message, stack: this.stack };
  };
  Test.prototype.complete = function(status, message, stack){
    if (this.phase_completed) return;
    this.phase_completed = true;
    if (this.status === TEST_STATUS.NOTRUN) {
      this.status = (status === undefined) ? TEST_STATUS.PASS : status;
    }
    if (message !== undefined && message !== null) this.message = message;
    if (stack !== undefined && stack !== null) this.stack = stack;
    run_cleanup(this);
    for (var i=0;i<result_callbacks.length;i++){ try { result_callbacks[i](this); } catch(e){} }
    maybe_finish();
  };
  Test.prototype.set_status = function(status, message, stack){
    this.status = status;
    this.message = message != null ? String(message) : null;
    this.stack = stack || null;
  };
  Test.prototype.done = function(){
    if (this.phase_completed) return;
    this.complete(this.status === TEST_STATUS.NOTRUN ? TEST_STATUS.PASS : this.status);
  };
  // step(func, this_obj, ...args): run a chunk of test code, catching assertion
  // failures and recording them against this test.
  Test.prototype.step = function(func, this_obj){
    if (this.phase_completed) return;
    if (this.status === TEST_STATUS.NOTRUN) this.status = TEST_STATUS.PASS;
    var args = Array.prototype.slice.call(arguments, 2);
    var ctx = (this_obj === undefined) ? this : this_obj;
    try {
      return func.apply(ctx, args);
    } catch(e){
      record_throw(this, e);
      return undefined;
    }
  };
  Test.prototype.step_func = function(func, this_obj){
    var test_this = this;
    var ctx = (this_obj === undefined) ? test_this : this_obj;
    return function(){
      var args = Array.prototype.slice.call(arguments);
      return Test.prototype.step.apply(test_this, [func, ctx].concat(args));
    };
  };
  Test.prototype.step_func_done = function(func, this_obj){
    var test_this = this;
    var ctx = (this_obj === undefined) ? test_this : this_obj;
    return function(){
      var args = Array.prototype.slice.call(arguments);
      if (func) Test.prototype.step.apply(test_this, [func, ctx].concat(args));
      test_this.done();
    };
  };
  Test.prototype.unreached_func = function(description){
    var test_this = this;
    return test_this.step_func(function(){
      assert_unreached(description);
    });
  };
  Test.prototype.step_timeout = function(func, timeout){
    var test_this = this;
    var args = Array.prototype.slice.call(arguments, 2);
    return setTimeout(test_this.step_func(function(){ return func.apply(test_this, args); }), timeout);
  };
  Test.prototype.add_cleanup = function(callback){ this.cleanup_callbacks.push(callback); };
  Test.prototype.force_timeout = function(){ this.set_status(TEST_STATUS.TIMEOUT, 'Test timed out'); };

  function record_throw(test, e){
    if (e instanceof OptionalFeatureUnsupportedError){
      test.set_status(TEST_STATUS.PRECONDITION_FAILED, e.message, e.stack);
    } else if (e instanceof AssertionError){
      test.set_status(TEST_STATUS.FAIL, e.message, e.stack);
    } else {
      var nm = (e && e.name) ? e.name : 'Error';
      var mg = (e && (e.message !== undefined)) ? e.message : String(e);
      test.set_status(TEST_STATUS.FAIL, nm + ': ' + mg, e && e.stack);
    }
  }

  function run_cleanup(test){
    for (var i=0;i<test.cleanup_callbacks.length;i++){
      try { test.cleanup_callbacks[i](); } catch(e){}
    }
    test.cleanup_callbacks = [];
  }

  // ---- public test entry points ------------------------------------------
  function test(func, name, properties){
    if (all_completed()) return;
    var t = new Test(name, properties);
    t.step(func, t, t);
    if (t.status === TEST_STATUS.NOTRUN) t.status = TEST_STATUS.PASS;
    t.complete(t.status);
    return t;
  }

  function async_test(func, name, properties){
    if (typeof func !== 'function'){ properties = name; name = func; func = null; }
    var t = new Test(name, properties);
    t.is_async = true;
    if (func){ t.step(func, t, t); }
    return t;
  }

  function promise_test(func, name, properties){
    var t = new Test(name, properties);
    t.is_async = true;
    if (t.status === TEST_STATUS.NOTRUN) t.status = TEST_STATUS.PASS;
    var donep;
    try {
      donep = Promise.resolve(t.step(function(){ return func.call(t, t); }, t, t));
    } catch(e){
      record_throw(t, e);
      t.complete(t.status);
      return t;
    }
    donep.then(function(){
      t.complete(t.status);
    }, function(reason){
      record_throw(t, reason);
      t.complete(t.status);
    });
    return t;
  }

  function promise_rejects_js(test, constructor, promise, description){
    return promise.then(
      test.step_func(function(){ assert_unreached('Should have rejected: ' + (description||'')); }),
      function(e){ assert_throws_js_impl(constructor, function(){ throw e; }, description); });
  }
  function promise_rejects_dom(test, type, promiseOrNode, descOrPromise, description){
    var promise = (typeof descOrPromise === 'object' && descOrPromise && descOrPromise.then) ? descOrPromise : promiseOrNode;
    return promise.then(
      test.step_func(function(){ assert_unreached('Should have rejected: ' + (description||'')); }),
      function(e){ assert_throws_dom_impl(type, function(){ throw e; }, description); });
  }
  function promise_rejects_exactly(test, exception, promise, description){
    return promise.then(
      test.step_func(function(){ assert_unreached('Should have rejected: ' + (description||'')); }),
      function(e){ assert_equals(e, exception, description); });
  }

  function setup(func_or_properties, maybe_properties){
    var props = func_or_properties;
    if (typeof func_or_properties === 'function'){
      try { func_or_properties(); } catch(e){
        harness_status.status = STATUS.ERROR;
        harness_status.message = String(e && (e.message||e));
      }
      props = maybe_properties;
    }
    if (props && typeof props === 'object'){
      if (props.explicit_done) settings.explicit_done = true;
      if (props.explicit_timeout) settings.explicit_timeout = true;
      if (props.single_test) { settings.single_test = true; single_test_mode = true; }
    }
  }
  var promise_setup = setup;

  function done(){
    if (phase_complete) return;
    settings.explicit_done = true;
    all_loaded = true;
    // close out any async tests still NOTRUN as PASS if they recorded no failure
    finish_all();
  }

  function add_completion_callback(cb){ completion_callbacks.push(cb); }
  function add_result_callback(cb){ result_callbacks.push(cb); }
  function add_start_callback(cb){ /* no visible reporter; ignore */ }

  function all_completed(){ return phase_complete; }

  function pending_async(){
    for (var i=0;i<tests.length;i++){
      if (tests[i].is_async && !tests[i].phase_completed && tests[i].status === TEST_STATUS.NOTRUN){
        return true;
      }
    }
    return false;
  }

  function maybe_finish(){
    if (phase_complete) return;
    if (settings.explicit_done && !all_loaded) return;
    if (!all_loaded) return;
    if (pending_async()) return;
    finish_all();
  }

  function finish_all(){
    if (phase_complete) return;
    phase_complete = true;
    for (var i=0;i<tests.length;i++){
      var t = tests[i];
      if (!t.phase_completed){
        if (t.is_async && t.status === TEST_STATUS.NOTRUN){
          t.status = TEST_STATUS.TIMEOUT;
          t.message = t.message || 'Test did not complete (no done())';
        } else if (t.status === TEST_STATUS.NOTRUN){
          t.status = TEST_STATUS.PASS;
        }
        t.phase_completed = true;
        run_cleanup(t);
      }
    }
    var summaries = [];
    for (var j=0;j<tests.length;j++){ summaries.push(tests[j].structured_clone()); }
    for (var k=0;k<completion_callbacks.length;k++){
      try { completion_callbacks[k](summaries, harness_status); } catch(e){}
    }
  }

  // Called by the Rust host after the load event + scheduler drain, to force
  // collection of results even if a test forgot done() or the doc had no load.
  function host_finalize(){
    all_loaded = true;
    finish_all();
  }

  // ---- assertions ---------------------------------------------------------
  function assert(expected_true, function_name, description, error, substitutions){
    if (expected_true !== true){
      throw new AssertionError(make_message(function_name, description, error, substitutions));
    }
  }

  function assert_true(actual, description){
    assert(actual === true, 'assert_true', description,
      'expected true got ${actual}', { actual: actual });
  }
  function assert_false(actual, description){
    assert(actual === false, 'assert_false', description,
      'expected false got ${actual}', { actual: actual });
  }
  function assert_equals(actual, expected, description){
    if (typeof actual !== typeof expected){
      assert(false, 'assert_equals', description,
        'expected (' + typeof expected + ') ${expected} but got (' + typeof actual + ') ${actual}',
        { expected: expected, actual: actual });
      return;
    }
    assert(same_value(actual, expected), 'assert_equals', description,
      'expected ${expected} but got ${actual}', { expected: expected, actual: actual });
  }
  function assert_not_equals(actual, expected, description){
    assert(!same_value(actual, expected), 'assert_not_equals', description,
      'got disallowed value ${actual}', { actual: actual });
  }
  function assert_in_array(actual, expected, description){
    var found = false;
    for (var i=0;i<expected.length;i++){ if (same_value(actual, expected[i])){ found = true; break; } }
    assert(found, 'assert_in_array', description,
      'value ${actual} not in array ${expected}', { actual: actual, expected: expected });
  }
  function assert_array_equals(actual, expected, description){
    assert(actual != null && typeof actual.length === 'number', 'assert_array_equals', description,
      'value is not array-like ${actual}', { actual: actual });
    assert(actual.length === expected.length, 'assert_array_equals', description,
      'lengths differ, expected array ${expected} length ${expected_len} got ${actual} length ${actual_len}',
      { expected: expected, actual: actual, expected_len: expected.length, actual_len: actual.length });
    for (var i=0;i<actual.length;i++){
      var has_a = Object.prototype.hasOwnProperty.call(actual, i);
      var has_e = Object.prototype.hasOwnProperty.call(expected, i);
      assert(has_a === has_e, 'assert_array_equals', description,
        'property ${i} missing in one array', { i: i });
      assert(same_value(actual[i], expected[i]), 'assert_array_equals', description,
        'property ${i} expected ${expected} got ${actual}',
        { i: i, expected: expected[i], actual: actual[i] });
    }
  }
  function assert_object_equals(actual, expected, description){
    deep_eq(actual, expected, [], description);
  }
  function deep_eq(actual, expected, stack, description){
    if (typeof actual !== 'object' || actual === null){
      assert(same_value(actual, expected), 'assert_object_equals', description,
        'expected ${expected} got ${actual}', { expected: expected, actual: actual });
      return;
    }
    for (var p in expected){
      assert(Object.prototype.hasOwnProperty.call(actual, p), 'assert_object_equals', description,
        'missing property ${p}', { p: p });
      deep_eq(actual[p], expected[p], stack.concat([p]), description);
    }
    for (var q in actual){
      assert(Object.prototype.hasOwnProperty.call(expected, q), 'assert_object_equals', description,
        'unexpected property ${q}', { q: q });
    }
  }
  function assert_approx_equals(actual, expected, epsilon, description){
    assert(typeof actual === 'number', 'assert_approx_equals', description,
      'expected a number got ${actual}', { actual: actual });
    assert(Math.abs(actual - expected) <= epsilon, 'assert_approx_equals', description,
      'expected ${expected} +/- ${epsilon} but got ${actual}',
      { expected: expected, epsilon: epsilon, actual: actual });
  }
  function assert_less_than(actual, expected, description){
    assert(actual < expected, 'assert_less_than', description,
      'expected a number less than ${expected} but got ${actual}', { expected: expected, actual: actual });
  }
  function assert_greater_than(actual, expected, description){
    assert(actual > expected, 'assert_greater_than', description,
      'expected a number greater than ${expected} but got ${actual}', { expected: expected, actual: actual });
  }
  function assert_less_than_equal(actual, expected, description){
    assert(actual <= expected, 'assert_less_than_equal', description,
      'expected <= ${expected} but got ${actual}', { expected: expected, actual: actual });
  }
  function assert_greater_than_equal(actual, expected, description){
    assert(actual >= expected, 'assert_greater_than_equal', description,
      'expected >= ${expected} but got ${actual}', { expected: expected, actual: actual });
  }
  function assert_between_exclusive(actual, lower, upper, description){
    assert(actual > lower && actual < upper, 'assert_between_exclusive', description,
      'expected in (${lower},${upper}) got ${actual}', { lower: lower, upper: upper, actual: actual });
  }
  function assert_regexp_match(actual, regexp, description){
    assert(regexp.test(actual), 'assert_regexp_match', description,
      'expected ${actual} to match ${regexp}', { actual: actual, regexp: String(regexp) });
  }
  function assert_class_string(object, class_string, description){
    var actual = Object.prototype.toString.call(object);
    var expected = '[object ' + class_string + ']';
    assert(actual === expected, 'assert_class_string', description,
      'expected ${expected} got ${actual}', { expected: expected, actual: actual });
  }
  function assert_own_property(object, property_name, description){
    assert(object != null && Object.prototype.hasOwnProperty.call(object, property_name),
      'assert_own_property', description,
      'expected property ${p}', { p: property_name });
  }
  function assert_not_own_property(object, property_name, description){
    assert(!(object != null && Object.prototype.hasOwnProperty.call(object, property_name)),
      'assert_not_own_property', description,
      'unexpected property ${p}', { p: property_name });
  }
  function assert_inherits(object, property_name, description){
    assert(object != null && !Object.prototype.hasOwnProperty.call(object, property_name) &&
           (property_name in object),
      'assert_inherits', description,
      'expected inherited property ${p}', { p: property_name });
  }
  function assert_idl_attribute(object, property_name, description){
    assert(object != null && (property_name in object), 'assert_idl_attribute', description,
      'expected IDL attribute ${p}', { p: property_name });
  }
  function assert_readonly(object, property_name, description){
    var initial = object[property_name];
    try {
      object[property_name] = initial + 'a';
      assert(same_value(object[property_name], initial), 'assert_readonly', description,
        'property ${p} is not readonly', { p: property_name });
    } finally {}
  }
  function assert_unreached(description){
    assert(false, 'assert_unreached', description, 'Reached unreachable code');
  }
  function assert_precondition(precondition, description){
    if (!precondition){ throw new OptionalFeatureUnsupportedError(description || 'precondition failed'); }
  }
  function assert_implements(condition, description){
    if (!condition){ throw new OptionalFeatureUnsupportedError(description || 'not implemented'); }
  }
  function assert_implements_optional(condition, description){
    if (!condition){ throw new OptionalFeatureUnsupportedError(description || 'optional not implemented'); }
  }

  // assert_throws_js: a constructor (TypeError, RangeError, ...) is expected.
  function assert_throws_js_impl(constructor, func, description){
    try {
      func.call(this);
    } catch(e){
      var ok = (e instanceof constructor) ||
               (e && e.name && constructor && constructor.name && e.name === constructor.name) ||
               (e && e.constructor && constructor && e.constructor.name === constructor.name);
      assert(ok, 'assert_throws_js', description,
        'expected ${ctor} but got ${actual}',
        { ctor: (constructor && constructor.name) || 'error', actual: (e && e.name) || String(e) });
      return;
    }
    assert(false, 'assert_throws_js', description,
      'expected ${ctor} but no exception thrown', { ctor: (constructor && constructor.name) || 'error' });
  }
  function assert_throws_js(constructor, func, description){
    return assert_throws_js_impl(constructor, func, description);
  }

  // assert_throws_dom: a DOMException name/code is expected.
  var DOM_CODE = {
    INDEX_SIZE_ERR:1, HIERARCHY_REQUEST_ERR:3, WRONG_DOCUMENT_ERR:4,
    INVALID_CHARACTER_ERR:5, NO_MODIFICATION_ALLOWED_ERR:7, NOT_FOUND_ERR:8,
    NOT_SUPPORTED_ERR:9, INUSE_ATTRIBUTE_ERR:10, INVALID_STATE_ERR:11,
    SYNTAX_ERR:12, INVALID_MODIFICATION_ERR:13, NAMESPACE_ERR:14,
    INVALID_ACCESS_ERR:15, SECURITY_ERR:18, NETWORK_ERR:19, ABORT_ERR:20,
    URL_MISMATCH_ERR:21, QUOTA_EXCEEDED_ERR:22, TIMEOUT_ERR:23,
    INVALID_NODE_TYPE_ERR:24, DATA_CLONE_ERR:25
  };
  var DOM_NAME_FOR_CODE = {
    1:'IndexSizeError', 3:'HierarchyRequestError', 4:'WrongDocumentError',
    5:'InvalidCharacterError', 7:'NoModificationAllowedError', 8:'NotFoundError',
    9:'NotSupportedError', 10:'InUseAttributeError', 11:'InvalidStateError',
    12:'SyntaxError', 13:'InvalidModificationError', 14:'NamespaceError',
    15:'InvalidAccessError', 18:'SecurityError', 19:'NetworkError', 20:'AbortError',
    21:'URLMismatchError', 22:'QuotaExceededError', 23:'TimeoutError',
    24:'InvalidNodeTypeError', 25:'DataCloneError'
  };
  function assert_throws_dom_impl(type, func, description){
    try {
      func.call(this);
    } catch(e){
      var wantName, wantCode;
      if (typeof type === 'number'){ wantCode = type; wantName = DOM_NAME_FOR_CODE[type]; }
      else if (typeof type === 'string'){
        if (DOM_CODE[type] !== undefined){ wantCode = DOM_CODE[type]; wantName = DOM_NAME_FOR_CODE[wantCode]; }
        else { wantName = type; }
      }
      var gotName = e && e.name;
      var gotCode = e && e.code;
      var ok = false;
      if (wantName && gotName === wantName) ok = true;
      if (!ok && wantCode !== undefined && gotCode === wantCode && gotCode !== 0) ok = true;
      assert(ok, 'assert_throws_dom', description,
        'expected DOMException ${want} but got ${got}',
        { want: (wantName || wantCode), got: (gotName || String(e)) });
      return;
    }
    assert(false, 'assert_throws_dom', description,
      'expected DOMException ${want} but no exception thrown', { want: type });
  }
  function assert_throws_dom(type, funcOrConstructor, descOrFunc, maybeDesc){
    // WPT signature variants: (type, func, desc) and (type, constructor, func, desc)
    if (typeof descOrFunc === 'function'){
      return assert_throws_dom_impl(type, descOrFunc, maybeDesc);
    }
    return assert_throws_dom_impl(type, funcOrConstructor, descOrFunc);
  }
  function assert_throws_exactly(exception, func, description){
    try { func.call(this); }
    catch(e){
      assert(same_value(e, exception), 'assert_throws_exactly', description,
        'expected exactly ${expected} got ${actual}', { expected: exception, actual: e });
      return;
    }
    assert(false, 'assert_throws_exactly', description, 'no exception thrown');
  }
  // legacy spelling
  function assert_throws(code, func, description){
    return assert_throws_dom_impl(code, func, description);
  }

  // ---- generate_tests / test_environment helpers --------------------------
  function generate_tests(func, args, properties){
    for (var i=0;i<args.length;i++){
      (function(a){
        var name = a[0];
        var rest = a.slice(1);
        test(function(){ func.apply(this, rest); }, name, properties);
      })(args[i]);
    }
  }

  // EventWatcher: a minimal version sufficient for simple wait_for usage.
  function EventWatcher(test, watchedNode, eventTypes){
    if (typeof eventTypes === 'string') eventTypes = [eventTypes];
    var waitingFor = null;
    function eventHandler(evt){
      if (!waitingFor){ return; }
      if (waitingFor.types.indexOf(evt.type) === -1){ return; }
      var w = waitingFor; waitingFor = null;
      w.resolve(evt);
    }
    for (var i=0;i<eventTypes.length;i++){
      try { watchedNode.addEventListener(eventTypes[i], eventHandler, false); } catch(e){}
    }
    this.wait_for = function(types){
      if (typeof types === 'string') types = [types];
      return new Promise(function(resolve, reject){
        waitingFor = { types: types, resolve: resolve, reject: reject };
      });
    };
  }

  // ---- expose -------------------------------------------------------------
  var api = {
    test: test, async_test: async_test, promise_test: promise_test,
    promise_setup: promise_setup, setup: setup, done: done,
    generate_tests: generate_tests,
    add_completion_callback: add_completion_callback,
    add_result_callback: add_result_callback,
    add_start_callback: add_start_callback,
    promise_rejects_js: promise_rejects_js,
    promise_rejects_dom: promise_rejects_dom,
    promise_rejects_exactly: promise_rejects_exactly,
    EventWatcher: EventWatcher,
    format_value: format_value,
    assert_true: assert_true, assert_false: assert_false,
    assert_equals: assert_equals, assert_not_equals: assert_not_equals,
    assert_in_array: assert_in_array,
    assert_array_equals: assert_array_equals,
    assert_object_equals: assert_object_equals,
    assert_approx_equals: assert_approx_equals,
    assert_less_than: assert_less_than, assert_greater_than: assert_greater_than,
    assert_less_than_equal: assert_less_than_equal,
    assert_greater_than_equal: assert_greater_than_equal,
    assert_between_exclusive: assert_between_exclusive,
    assert_regexp_match: assert_regexp_match,
    assert_class_string: assert_class_string,
    assert_own_property: assert_own_property,
    assert_not_own_property: assert_not_own_property,
    assert_inherits: assert_inherits,
    assert_idl_attribute: assert_idl_attribute,
    assert_readonly: assert_readonly,
    assert_unreached: assert_unreached,
    assert_precondition: assert_precondition,
    assert_implements: assert_implements,
    assert_implements_optional: assert_implements_optional,
    assert_throws_js: assert_throws_js,
    assert_throws_dom: assert_throws_dom,
    assert_throws_exactly: assert_throws_exactly,
    assert_throws: assert_throws
  };
  for (var key in api){
    try { globalThis[key] = api[key]; } catch(e){}
  }

  // Internal harness object the Rust host reads.
  globalThis.__wpt = {
    TEST_STATUS: TEST_STATUS,
    STATUS: STATUS,
    tests: tests,
    host_finalize: host_finalize,
    harness_status: harness_status,
    serialize: function(){
      host_finalize();
      var out = { harness_status: harness_status.status, tests: [] };
      var names = ['PASS','FAIL','TIMEOUT','NOTRUN','PRECONDITION_FAILED'];
      for (var i=0;i<tests.length;i++){
        var t = tests[i];
        out.tests.push({
          name: t.name == null ? ('test #' + (i+1)) : String(t.name),
          status: t.status,
          status_name: names[t.status] || String(t.status),
          message: t.message == null ? '' : String(t.message)
        });
      }
      return JSON.stringify(out);
    }
  };

  // Auto-completion on window load (the common case for tests that don't call
  // done()). The host also calls host_finalize() as a backstop.
  try {
    if (typeof window !== 'undefined' && window.addEventListener){
      window.addEventListener('load', function(){ all_loaded = true; maybe_finish(); }, false);
    }
  } catch(e){}
})();
"#;
