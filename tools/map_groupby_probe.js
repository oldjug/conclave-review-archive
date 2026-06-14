console.log(
  "MAP_GB_PROBE",
  JSON.stringify({
    typeofMap: typeof Map,
    typeofMapGroupBy: typeof Map.groupBy,
    hasOwn: Object.prototype.hasOwnProperty.call(Map, "groupBy"),
    ownNames: Object.getOwnPropertyNames(Map).slice(0, 20),
  })
);
