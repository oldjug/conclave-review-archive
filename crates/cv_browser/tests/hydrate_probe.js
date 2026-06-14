// Diagnostic probe: figure out how far Next.js App Router bootstrap got.
function rpt(k, v) { console.log("PROBE " + k + " = " + v); }

try { rpt("window.next", typeof window.next + " / " + (window.next ? Object.keys(window.next).join(",") : "undefined")); } catch(e){ console.error("hydrate_probe: error checking window.next:", e); rpt("window.next ERR", e); }

// webpack chunk global
try {
  var wk = Object.keys(window).filter(function(k){ return /webpackChunk/.test(k); });
  rpt("webpackChunk keys", wk.join(","));
  wk.forEach(function(k){
    var a = window[k];
    rpt(k + ".length", (a && a.length));
    rpt(k + ".push name", (a && a.push && a.push.name));
  });
} catch(e){ console.error("hydrate_probe: error checking webpack chunks:", e); rpt("webpack ERR", e); }

// __next_f
try {
  rpt("__next_f type", typeof window.__next_f);
  rpt("__next_f.length", window.__next_f && window.__next_f.length);
  rpt("__next_f.push name", window.__next_f && window.__next_f.push && window.__next_f.push.name);
} catch(e){ console.error("hydrate_probe: error checking __next_f:", e); rpt("__next_f ERR", e); }

// Did React mount anything? Check for fiber roots / data-reactroot
try {
  var html = document.documentElement;
  var keys = [];
  for (var k in html) { if (/react|fiber|__reactContainer/i.test(k)) keys.push(k); }
  rpt("documentElement react keys", keys.join(",") || "(none)");
} catch(e){ console.error("hydrate_probe: error checking React fiber keys:", e); rpt("fiber ERR", e); }

// Is there a __webpack_require__ leaked? (usually not — module scoped)
try { rpt("__webpack_require__", typeof window.__webpack_require__); } catch(e){}
// `_N_E` is assigned by the main-app entry runtime (`_N_E=e.O()`); if it
// exists at all (even undefined value) the entry runtime FUNCTION ran.
try { rpt("_N_E in window", ('_N_E' in window) + " / typeof=" + typeof _N_E); } catch(e){ console.error("hydrate_probe: error checking _N_E:", e); rpt("_N_E ERR", e); }
// Dump the chunk-id arrays in push order to see ordering + whether 2971/2117
// (the entry's required deps) are present.
try {
  var arr = window.webpackChunk_N_E || [];
  var ids = [];
  for (var idx=0; idx<arr.length; idx++) { try { ids.push(JSON.stringify(arr[idx][0])); } catch(e){ console.error("hydrate_probe: error stringifying chunk id at index " + idx + ":", e); ids.push("?"); } }
  rpt("chunk id arrays", ids.join(" "));
} catch(e){ console.error("hydrate_probe: error processing chunk id arrays:", e); rpt("chunkids ERR", e); }

// Count elements (did hydration add/replace anything?)
try { rpt("total elements", document.getElementsByTagName("*").length); } catch(e){}

// Is requestAnimationFrame being used (particles)?
try { rpt("rafQueued", (window.__rafCount===undefined?"n/a":window.__rafCount)); } catch(e){}

// Look for the particles canvas
try {
  var cs = document.getElementsByTagName("canvas");
  rpt("canvas count", cs.length);
} catch(e){}
