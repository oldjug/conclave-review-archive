//! `Temporal` — the modern date/time API (TC39 Temporal proposal, Stage 3,
//! shipping in V8). This module implements the **core** of the proposal with
//! REAL calendar/time arithmetic (no fixed-value stubs):
//!
//!   * `Temporal.Now` — `instant()`, `plainDateISO()`, `plainTimeISO()`,
//!     `plainDateTimeISO()`, `zonedDateTimeISO()` (UTC zone), `timeZoneId()`.
//!   * `Temporal.PlainDate` — year/month/day; `add`/`subtract` (calendar
//!     arithmetic with month-overflow normalization + day `constrain`/`reject`),
//!     `with`, `until`/`since` (calendar Duration via the spec's
//!     DifferenceISODate, default `largestUnit: 'auto' → 'day'`), `compare`
//!     (static, -1/0/1), `equals`, `toString` (RFC 9557 ISO `YYYY-MM-DD`),
//!     `dayOfWeek`, `daysInMonth`, `daysInYear`, `inLeapYear`.
//!   * `Temporal.PlainTime` — hour..nanosecond; `add`/`subtract` (wrap mod
//!     24h), `with`, `toString` (`HH:MM:SS[.fffffffff]`), `compare`, `equals`.
//!   * `Temporal.PlainDateTime` — combines the two; `add`/`subtract` (time
//!     overflow carries into the date), `with`, `compare`, `equals`,
//!     `toPlainDate`/`toPlainTime`, `toString`.
//!   * `Temporal.Instant` — exact time as epoch nanoseconds (i128); `add`/
//!     `subtract` (time-unit Duration), `until`/`since` (nanosecond Duration),
//!     `epochMilliseconds`/`epochNanoseconds`, `compare`, `equals`, `toString`
//!     (`...Z`).
//!   * `Temporal.Duration` — years..nanoseconds; `add`/`subtract`, `negated`,
//!     `abs`, `with`, `total({unit})` for time units, `blank`, `sign`,
//!     `toString` (ISO-8601 `P…T…`).
//!   * `Temporal.ZonedDateTime` — **UTC-offset core**: wraps an `Instant` +
//!     time-zone id; UTC + fixed `±HH:MM` offsets are real; named IANA zones
//!     resolve to UTC (documented followup — see module footer). Has
//!     `epochNanoseconds`, `toInstant`, `toPlainDateTime`, `add`/`subtract`,
//!     `compare`, `equals`, `toString`.
//!
//! Spec references are cited inline (`tc39.es/proposal-temporal`). The
//! calendar is ISO-8601 only; other calendar systems are a documented followup.
//!
//! Representation: each Temporal value is a `Value::Object` carrying a brand
//! key (`_temporalKind`) plus its ISO fields as `Value::Number`s (Instant/
//! ZonedDateTime carry epoch-ns as a decimal string to preserve full i128
//! range past f64's 2^53). Methods read `this` via `current_native_this` —
//! the same idiom as `Date`. Values are immutable (every operation returns a
//! fresh object).

use crate::interp::{
    current_native_this, make_temporal_error, native_fn_with_interp, Interp, JsError, Value,
};
use crate::ordered::OrderedMap as HashMap;
use std::cell::RefCell;
use std::rc::Rc;

// ============================================================
// Pure calendar/time math — no `Value`, fully unit-testable.
// ============================================================

/// ISO-8601 leap year (proleptic Gregorian). tc39 `MathematicalDaysInYear`.
pub fn is_iso_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Days in `month` (1..=12) of `year`. tc39 `ISODaysInMonth`.
pub fn iso_days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_iso_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

pub fn days_in_year(year: i64) -> i64 {
    if is_iso_leap_year(year) {
        366
    } else {
        365
    }
}

/// Howard Hinnant's `days_from_civil`: days since the Unix epoch
/// (1970-01-01) for a proleptic-Gregorian (y, m, d). tc39 `ISODateToEpochDays`
/// uses the same well-known algorithm. Valid for the full i64 range.
pub fn epoch_days(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// Inverse of [`epoch_days`] — Hinnant `civil_from_days`.
pub fn civil_from_epoch_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d)
}

/// Day of week, ISO numbering Mon=1..Sun=7. tc39 `ISODayOfWeek`.
pub fn day_of_week(year: i64, month: i64, day: i64) -> i64 {
    // 1970-01-01 was a Thursday (=4). epoch_days(1970,1,1) == 0.
    let ed = epoch_days(year, month, day);
    ((ed.rem_euclid(7)) + 3).rem_euclid(7) + 1
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overflow {
    Constrain,
    Reject,
}

/// Validate (and possibly clamp) an ISO date per the requested overflow.
/// tc39 `RegulateISODate`. In `Constrain`, month is clamped to 1..=12 and day
/// to 1..=daysInMonth; in `Reject`, an out-of-range value throws (here:
/// returns `None`).
pub fn regulate_iso_date(
    year: i64,
    month: i64,
    day: i64,
    overflow: Overflow,
) -> Option<(i64, i64, i64)> {
    match overflow {
        Overflow::Constrain => {
            let m = month.clamp(1, 12);
            let d = day.clamp(1, iso_days_in_month(year, m));
            Some((year, m, d))
        }
        Overflow::Reject => {
            if !(1..=12).contains(&month) {
                return None;
            }
            if day < 1 || day > iso_days_in_month(year, month) {
                return None;
            }
            Some((year, month, day))
        }
    }
}

pub fn regulate_iso_time(
    h: i64,
    mi: i64,
    s: i64,
    ms: i64,
    us: i64,
    ns: i64,
    overflow: Overflow,
) -> Option<(i64, i64, i64, i64, i64, i64)> {
    match overflow {
        Overflow::Constrain => Some((
            h.clamp(0, 23),
            mi.clamp(0, 59),
            s.clamp(0, 59),
            ms.clamp(0, 999),
            us.clamp(0, 999),
            ns.clamp(0, 999),
        )),
        Overflow::Reject => {
            if !(0..=23).contains(&h)
                || !(0..=59).contains(&mi)
                || !(0..=59).contains(&s)
                || !(0..=999).contains(&ms)
                || !(0..=999).contains(&us)
                || !(0..=999).contains(&ns)
            {
                return None;
            }
            Some((h, mi, s, ms, us, ns))
        }
    }
}

/// Add a calendar date-duration (years, months, weeks, days) to an ISO date.
/// tc39 `AddISODate` / `BalanceISOYearMonth`: years+months are applied first
/// (month overflow normalized into years), the day is clamped to the new
/// month per `overflow`, then weeks*7 + days are added as whole days.
/// Returns `None` if `Reject` overflow rejects.
pub fn add_iso_date(
    year: i64,
    month: i64,
    day: i64,
    years: i64,
    months: i64,
    weeks: i64,
    days: i64,
    overflow: Overflow,
) -> Option<(i64, i64, i64)> {
    // BalanceISOYearMonth: y += years; total months -> normalize to 1..=12.
    let mut y = year + years;
    // month is 1..=12; add `months`, normalize into year.
    let total = (month - 1) + months; // zero-based month index, may be negative
    y += total.div_euclid(12);
    let m = total.rem_euclid(12) + 1;
    // Constrain/reject the day within the (possibly new) month.
    let (y, m, d) = regulate_iso_date(y, m, day, overflow)?;
    // Add whole days (weeks*7 + days) via epoch-day arithmetic.
    let ed = epoch_days(y, m, d) + weeks * 7 + days;
    Some(civil_from_epoch_days(ed))
}

/// Calendar difference `later - earlier` (both must be normalized ISO dates),
/// expressed in (years, months, weeks, days) up to `largest`. tc39
/// `DifferenceISODate`. `weeks` is only produced when largestUnit == Week;
/// otherwise its slot is 0. Result is signed (negative if `b` < `a`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DateUnit {
    Year,
    Month,
    Week,
    Day,
}

pub fn difference_iso_date(
    ay: i64,
    am: i64,
    ad: i64,
    by: i64,
    bm: i64,
    bd: i64,
    largest: DateUnit,
) -> (i64, i64, i64, i64) {
    // Sign: are we going forward (a <= b) or backward?
    let cmp = compare_iso_date(ay, am, ad, by, bm, bd);
    if cmp == 0 {
        return (0, 0, 0, 0);
    }
    // Work with (start, end) in forward order, then re-sign.
    let (sign, sy, sm, sd, ey, em, ed) = if cmp < 0 {
        (1i64, ay, am, ad, by, bm, bd)
    } else {
        (-1i64, by, bm, bd, ay, am, ad)
    };

    match largest {
        DateUnit::Day | DateUnit::Week => {
            let total_days = epoch_days(ey, em, ed) - epoch_days(sy, sm, sd);
            if matches!(largest, DateUnit::Week) {
                let weeks = total_days / 7;
                let days = total_days % 7;
                (0, 0, sign * weeks, sign * days)
            } else {
                (0, 0, 0, sign * total_days)
            }
        }
        DateUnit::Year | DateUnit::Month => {
            // Count whole years, then whole months, then the day remainder —
            // the spec's calendar-aware loop (intermediate must not overshoot).
            let mut years = ey - sy;
            // Tentative intermediate after adding `years` years.
            let mut mid = add_iso_date(sy, sm, sd, years, 0, 0, 0, Overflow::Constrain).unwrap();
            if compare_iso_date(mid.0, mid.1, mid.2, ey, em, ed) > 0 {
                years -= 1;
                mid = add_iso_date(sy, sm, sd, years, 0, 0, 0, Overflow::Constrain).unwrap();
            }
            // Now add whole months from `mid` toward end.
            let mut months = 0i64;
            loop {
                let next =
                    add_iso_date(mid.0, mid.1, mid.2, 0, months + 1, 0, 0, Overflow::Constrain)
                        .unwrap();
                if compare_iso_date(next.0, next.1, next.2, ey, em, ed) > 0 {
                    break;
                }
                months += 1;
            }
            let after_months =
                add_iso_date(mid.0, mid.1, mid.2, 0, months, 0, 0, Overflow::Constrain).unwrap();
            let days =
                epoch_days(ey, em, ed) - epoch_days(after_months.0, after_months.1, after_months.2);
            if matches!(largest, DateUnit::Month) {
                // Fold years into months.
                (0, sign * (years * 12 + months), 0, sign * days)
            } else {
                (sign * years, sign * months, 0, sign * days)
            }
        }
    }
}

/// -1/0/1 ordering of two ISO dates. tc39 `CompareISODate`.
pub fn compare_iso_date(ay: i64, am: i64, ad: i64, by: i64, bm: i64, bd: i64) -> i64 {
    match (ay, am, ad).cmp(&(by, bm, bd)) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

/// Nanoseconds within a day for a PlainTime. 0..86_400_000_000_000.
pub fn time_to_nanos(h: i64, mi: i64, s: i64, ms: i64, us: i64, ns: i64) -> i64 {
    ((((h * 60 + mi) * 60 + s) * 1000 + ms) * 1000 + us) * 1000 + ns
}

/// Inverse of [`time_to_nanos`], plus the day-overflow (can be negative).
/// tc39 `BalanceTime`. Returns (days, h, mi, s, ms, us, ns).
pub fn balance_time(total_ns: i128) -> (i64, i64, i64, i64, i64, i64, i64) {
    const NS_PER_DAY: i128 = 86_400_000_000_000;
    let days = total_ns.div_euclid(NS_PER_DAY) as i64;
    let mut r = total_ns.rem_euclid(NS_PER_DAY);
    let ns = (r % 1000) as i64;
    r /= 1000;
    let us = (r % 1000) as i64;
    r /= 1000;
    let ms = (r % 1000) as i64;
    r /= 1000;
    let s = (r % 60) as i64;
    r /= 60;
    let mi = (r % 60) as i64;
    r /= 60;
    let h = r as i64;
    (days, h, mi, s, ms, us, ns)
}

/// Format an ISO time, omitting the fractional part when zero, else emitting
/// the minimum number of groups of 3 digits (3/6/9). tc39
/// `FormatTimeString` / `TemporalTimeToString`.
pub fn format_iso_time(h: i64, mi: i64, s: i64, ms: i64, us: i64, ns: i64) -> String {
    let mut out = format!("{:02}:{:02}:{:02}", h, mi, s);
    let frac = ms * 1_000_000 + us * 1000 + ns; // nanoseconds 0..=999_999_999
    if frac != 0 {
        // precision "auto": emit 9 digits then trim ALL trailing zeros (tc39
        // FormatFractionalSeconds — e.g. 500_000_000ns → ".5", not ".500").
        let mut digits = format!("{:09}", frac);
        while digits.ends_with('0') {
            digits.pop();
        }
        out.push('.');
        out.push_str(&digits);
    }
    out
}

/// Parse `±HH:MM`, `±HHMM`, `±HH`, or `Z`/`+00:00` style UTC offset → minutes.
/// tc39 `ParseTimeZoneOffsetString` (minute precision subset).
pub fn parse_offset_minutes(s: &str) -> Option<i64> {
    let s = s.trim();
    if s == "Z" || s == "z" {
        return Some(0);
    }
    let mut chars = s.chars();
    let (sign, sign_len) = match chars.next() {
        Some('+') => (1i64, 1usize),
        Some('-') => (-1, 1),
        // U+2212 MINUS SIGN (3 UTF-8 bytes) is accepted by tc39's grammar.
        Some('\u{2212}') => (-1, '\u{2212}'.len_utf8()),
        _ => return None,
    };
    let rest = &s[sign_len..];
    let digits: String = rest.chars().filter(|c| c.is_ascii_digit()).collect();
    let (h, m) = match digits.len() {
        2 => (digits.parse::<i64>().ok()?, 0),
        4 => (
            digits[..2].parse::<i64>().ok()?,
            digits[2..].parse::<i64>().ok()?,
        ),
        _ => return None,
    };
    if h > 23 || m > 59 {
        return None;
    }
    Some(sign * (h * 60 + m))
}

// ============================================================
// JS value <-> Temporal-fields glue
// ============================================================

const NS_PER_DAY_I128: i128 = 86_400_000_000_000;

fn range_err(msg: &str) -> JsError {
    make_temporal_error("RangeError", msg)
}
fn type_err(msg: &str) -> JsError {
    make_temporal_error("TypeError", msg)
}

/// Read `this` (native receiver) as a Temporal object of the given brand.
fn brand_this(kind: &str) -> Result<Rc<RefCell<HashMap<String, Value>>>, JsError> {
    let nt = current_native_this();
    if let Value::Object(o) = &nt {
        if let Some(Value::String(k)) = o.borrow().get("_temporalKind") {
            if &**k == kind {
                return Ok(o.clone());
            }
        }
    }
    Err(type_err(&format!("this is not a Temporal.{kind}")))
}

fn obj_num(o: &Rc<RefCell<HashMap<String, Value>>>, key: &str) -> i64 {
    match o.borrow().get(key) {
        Some(Value::Number(n)) => *n as i64,
        _ => 0,
    }
}

fn obj_str(o: &Rc<RefCell<HashMap<String, Value>>>, key: &str) -> Option<String> {
    match o.borrow().get(key) {
        Some(Value::String(s)) => Some((**s).to_string()),
        _ => None,
    }
}

/// Read a field from a duration-like / fields-like JS argument. Missing → def.
/// Truncates toward zero (ToIntegerOrInfinity-ish). NaN → error per spec, but
/// we keep it simple: NaN/non-finite is treated as the default.
fn arg_field(v: &Value, key: &str, def: i64) -> i64 {
    if let Value::Object(o) = v {
        if let Some(field) = o.borrow().get(key) {
            let n = field.to_number();
            if n.is_finite() {
                return n.trunc() as i64;
            }
        }
    }
    def
}

fn arg_has_any(v: &Value, keys: &[&str]) -> bool {
    if let Value::Object(o) = v {
        let b = o.borrow();
        keys.iter().any(|k| b.contains_key(*k))
    } else {
        false
    }
}

/// Extract the `overflow` option from an options object (arg index `idx`).
fn read_overflow(args: &[Value], idx: usize) -> Result<Overflow, JsError> {
    if let Some(Value::Object(o)) = args.get(idx) {
        if let Some(Value::String(s)) = o.borrow().get("overflow") {
            return match &**s {
                "constrain" => Ok(Overflow::Constrain),
                "reject" => Ok(Overflow::Reject),
                other => Err(range_err(&format!("invalid overflow option: {other}"))),
            };
        }
    }
    Ok(Overflow::Constrain)
}

// ----- object builders -----

fn mk(kind: &str, fields: &[(&str, Value)]) -> Value {
    let mut m: HashMap<String, Value> = HashMap::new();
    m.insert("_temporalKind".into(), Value::str(kind.to_string()));
    for (k, v) in fields {
        m.insert((*k).into(), v.clone());
    }
    let o = Rc::new(RefCell::new(m));
    attach_methods(kind, &o);
    Value::Object(o)
}

fn num(n: i64) -> Value {
    Value::Number(n as f64)
}

fn mk_plain_date(y: i64, m: i64, d: i64) -> Value {
    mk(
        "PlainDate",
        &[
            ("isoYear", num(y)),
            ("isoMonth", num(m)),
            ("isoDay", num(d)),
        ],
    )
}

fn mk_plain_time(h: i64, mi: i64, s: i64, ms: i64, us: i64, ns: i64) -> Value {
    mk(
        "PlainTime",
        &[
            ("isoHour", num(h)),
            ("isoMinute", num(mi)),
            ("isoSecond", num(s)),
            ("isoMillisecond", num(ms)),
            ("isoMicrosecond", num(us)),
            ("isoNanosecond", num(ns)),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn mk_plain_date_time(
    y: i64,
    mo: i64,
    d: i64,
    h: i64,
    mi: i64,
    s: i64,
    ms: i64,
    us: i64,
    ns: i64,
) -> Value {
    mk(
        "PlainDateTime",
        &[
            ("isoYear", num(y)),
            ("isoMonth", num(mo)),
            ("isoDay", num(d)),
            ("isoHour", num(h)),
            ("isoMinute", num(mi)),
            ("isoSecond", num(s)),
            ("isoMillisecond", num(ms)),
            ("isoMicrosecond", num(us)),
            ("isoNanosecond", num(ns)),
        ],
    )
}

fn mk_instant(epoch_ns: i128) -> Value {
    mk(
        "Instant",
        &[("epochNs", Value::str(epoch_ns.to_string()))],
    )
}

fn mk_zoned(epoch_ns: i128, tz: &str, offset_min: i64) -> Value {
    mk(
        "ZonedDateTime",
        &[
            ("epochNs", Value::str(epoch_ns.to_string())),
            ("timeZone", Value::str(tz.to_string())),
            ("offsetMinutes", num(offset_min)),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn mk_duration(
    years: i64,
    months: i64,
    weeks: i64,
    days: i64,
    hours: i64,
    minutes: i64,
    seconds: i64,
    millis: i64,
    micros: i64,
    nanos: i64,
) -> Value {
    mk(
        "Duration",
        &[
            ("years", num(years)),
            ("months", num(months)),
            ("weeks", num(weeks)),
            ("days", num(days)),
            ("hours", num(hours)),
            ("minutes", num(minutes)),
            ("seconds", num(seconds)),
            ("milliseconds", num(millis)),
            ("microseconds", num(micros)),
            ("nanoseconds", num(nanos)),
        ],
    )
}

fn read_instant_ns(o: &Rc<RefCell<HashMap<String, Value>>>) -> i128 {
    obj_str(o, "epochNs")
        .and_then(|s| s.parse::<i128>().ok())
        .unwrap_or(0)
}

// ============================================================
// Method attachment
// ============================================================

fn attach_methods(kind: &str, o: &Rc<RefCell<HashMap<String, Value>>>) {
    match kind {
        "PlainDate" => attach_plain_date_methods(o),
        "PlainTime" => attach_plain_time_methods(o),
        "PlainDateTime" => attach_plain_date_time_methods(o),
        "Instant" => attach_instant_methods(o),
        "Duration" => attach_duration_methods(o),
        "ZonedDateTime" => attach_zoned_methods(o),
        _ => {}
    }
}

macro_rules! meth {
    ($map:expr, $name:literal, $f:expr) => {
        $map.borrow_mut()
            .insert($name.to_string(), native_fn_with_interp($name, $f));
    };
}

fn pd_fields(o: &Rc<RefCell<HashMap<String, Value>>>) -> (i64, i64, i64) {
    (
        obj_num(o, "isoYear"),
        obj_num(o, "isoMonth"),
        obj_num(o, "isoDay"),
    )
}

fn pt_fields(o: &Rc<RefCell<HashMap<String, Value>>>) -> (i64, i64, i64, i64, i64, i64) {
    (
        obj_num(o, "isoHour"),
        obj_num(o, "isoMinute"),
        obj_num(o, "isoSecond"),
        obj_num(o, "isoMillisecond"),
        obj_num(o, "isoMicrosecond"),
        obj_num(o, "isoNanosecond"),
    )
}

fn attach_plain_date_methods(o: &Rc<RefCell<HashMap<String, Value>>>) {
    // Read-only getters realized as data props (year/month/day plus derived).
    {
        let (y, m, d) = pd_fields(o);
        let mut b = o.borrow_mut();
        b.insert("year".into(), num(y));
        b.insert("month".into(), num(m));
        b.insert("day".into(), num(d));
        b.insert("dayOfWeek".into(), num(day_of_week(y, m, d)));
        b.insert("daysInMonth".into(), num(iso_days_in_month(y, m)));
        b.insert("daysInYear".into(), num(days_in_year(y)));
        b.insert("monthsInYear".into(), num(12));
        b.insert("inLeapYear".into(), Value::Bool(is_iso_leap_year(y)));
        b.insert(
            "monthCode".into(),
            Value::str(format!("M{:02}", m)),
        );
    }

    meth!(o, "toString", |_i, _a| {
        let o = brand_this("PlainDate")?;
        let (y, m, d) = pd_fields(&o);
        Ok(Value::str(format_iso_date(y, m, d)))
    });
    meth!(o, "toJSON", |_i, _a| {
        let o = brand_this("PlainDate")?;
        let (y, m, d) = pd_fields(&o);
        Ok(Value::str(format_iso_date(y, m, d)))
    });
    meth!(o, "add", |_i, a: Vec<Value>| {
        let o = brand_this("PlainDate")?;
        let (y, m, d) = pd_fields(&o);
        let dur = a.first().cloned().unwrap_or(Value::Undefined);
        let ov = read_overflow(&a, 1)?;
        plain_date_add(y, m, d, &dur, 1, ov)
    });
    meth!(o, "subtract", |_i, a: Vec<Value>| {
        let o = brand_this("PlainDate")?;
        let (y, m, d) = pd_fields(&o);
        let dur = a.first().cloned().unwrap_or(Value::Undefined);
        let ov = read_overflow(&a, 1)?;
        plain_date_add(y, m, d, &dur, -1, ov)
    });
    meth!(o, "with", |_i, a: Vec<Value>| {
        let o = brand_this("PlainDate")?;
        let (y, m, d) = pd_fields(&o);
        let arg = a.first().cloned().unwrap_or(Value::Undefined);
        if !matches!(arg, Value::Object(_)) {
            return Err(type_err("with() requires a fields object"));
        }
        let ov = read_overflow(&a, 1)?;
        let ny = arg_field(&arg, "year", y);
        let nm = arg_field(&arg, "month", m);
        let nd = arg_field(&arg, "day", d);
        match regulate_iso_date(ny, nm, nd, ov) {
            Some((ry, rm, rd)) => Ok(mk_plain_date(ry, rm, rd)),
            None => Err(range_err("PlainDate.with: date out of range")),
        }
    });
    meth!(o, "until", |_i, a: Vec<Value>| {
        let o = brand_this("PlainDate")?;
        let (ay, am, ad) = pd_fields(&o);
        let (by, bm, bd) = other_plain_date(&a)?;
        let largest = read_date_largest_unit(&a, 1, DateUnit::Day)?;
        let (yy, mm, ww, dd) = difference_iso_date(ay, am, ad, by, bm, bd, largest);
        Ok(mk_duration(yy, mm, ww, dd, 0, 0, 0, 0, 0, 0))
    });
    meth!(o, "since", |_i, a: Vec<Value>| {
        let o = brand_this("PlainDate")?;
        let (ay, am, ad) = pd_fields(&o);
        let (by, bm, bd) = other_plain_date(&a)?;
        let largest = read_date_largest_unit(&a, 1, DateUnit::Day)?;
        // since(b) = -(b.until(a)) per spec; compute b - a then negate.
        let (yy, mm, ww, dd) = difference_iso_date(by, bm, bd, ay, am, ad, largest);
        Ok(mk_duration(yy, mm, ww, dd, 0, 0, 0, 0, 0, 0))
    });
    meth!(o, "equals", |_i, a: Vec<Value>| {
        let o = brand_this("PlainDate")?;
        let (ay, am, ad) = pd_fields(&o);
        let (by, bm, bd) = other_plain_date(&a)?;
        Ok(Value::Bool(compare_iso_date(ay, am, ad, by, bm, bd) == 0))
    });
    meth!(o, "toPlainDateTime", |_i, a: Vec<Value>| {
        let o = brand_this("PlainDate")?;
        let (y, m, d) = pd_fields(&o);
        // optional PlainTime arg
        let (h, mi, s, ms, us, ns) = if let Some(Value::Object(t)) = a.first() {
            if matches!(t.borrow().get("_temporalKind"), Some(Value::String(k)) if &**k == "PlainTime")
            {
                pt_fields(t)
            } else {
                (0, 0, 0, 0, 0, 0)
            }
        } else {
            (0, 0, 0, 0, 0, 0)
        };
        Ok(mk_plain_date_time(y, m, d, h, mi, s, ms, us, ns))
    });
}

/// Add/subtract a duration-like to a PlainDate. `sign` is +1 (add) / -1 (sub).
fn plain_date_add(
    y: i64,
    m: i64,
    d: i64,
    dur: &Value,
    sign: i64,
    overflow: Overflow,
) -> Result<Value, JsError> {
    // Accept Duration object / duration-like / ISO string.
    let (years, months, weeks, days) = read_date_duration(dur)?;
    match add_iso_date(
        y,
        m,
        d,
        sign * years,
        sign * months,
        sign * weeks,
        sign * days,
        overflow,
    ) {
        Some((ny, nm, nd)) => Ok(mk_plain_date(ny, nm, nd)),
        None => Err(range_err("PlainDate.add: result out of range (overflow:reject)")),
    }
}

/// Read (years, months, weeks, days) from a Duration value / duration-like /
/// ISO 8601 duration string. Sub-day units are read but ignored for PlainDate.
fn read_date_duration(v: &Value) -> Result<(i64, i64, i64, i64), JsError> {
    match v {
        Value::String(s) => {
            let d = parse_iso_duration(s)
                .ok_or_else(|| range_err("invalid ISO 8601 duration string"))?;
            Ok((d.0, d.1, d.2, d.3))
        }
        Value::Object(_) => Ok((
            arg_field(v, "years", 0),
            arg_field(v, "months", 0),
            arg_field(v, "weeks", 0),
            arg_field(v, "days", 0),
        )),
        _ => Err(type_err("expected a Duration or duration-like object")),
    }
}

/// Read all 10 duration components from a Duration value / duration-like / ISO
/// string.
type Dur10 = (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64);
fn read_full_duration(v: &Value) -> Result<Dur10, JsError> {
    match v {
        Value::String(s) => {
            parse_iso_duration(s).ok_or_else(|| range_err("invalid ISO 8601 duration string"))
        }
        Value::Object(_) => Ok((
            arg_field(v, "years", 0),
            arg_field(v, "months", 0),
            arg_field(v, "weeks", 0),
            arg_field(v, "days", 0),
            arg_field(v, "hours", 0),
            arg_field(v, "minutes", 0),
            arg_field(v, "seconds", 0),
            arg_field(v, "milliseconds", 0),
            arg_field(v, "microseconds", 0),
            arg_field(v, "nanoseconds", 0),
        )),
        _ => Err(type_err("expected a Duration or duration-like object")),
    }
}

fn other_plain_date(a: &[Value]) -> Result<(i64, i64, i64), JsError> {
    match a.first() {
        Some(Value::Object(o))
            if matches!(o.borrow().get("_temporalKind"), Some(Value::String(k)) if &**k == "PlainDate") =>
        {
            Ok(pd_fields(o))
        }
        Some(Value::String(s)) => parse_iso_date_str(s).ok_or_else(|| range_err("invalid PlainDate string")),
        _ => Err(type_err("expected a Temporal.PlainDate")),
    }
}

fn read_date_largest_unit(
    a: &[Value],
    idx: usize,
    def: DateUnit,
) -> Result<DateUnit, JsError> {
    if let Some(Value::Object(o)) = a.get(idx) {
        if let Some(Value::String(s)) = o.borrow().get("largestUnit") {
            return match &**s {
                "auto" => Ok(def),
                "year" | "years" => Ok(DateUnit::Year),
                "month" | "months" => Ok(DateUnit::Month),
                "week" | "weeks" => Ok(DateUnit::Week),
                "day" | "days" => Ok(DateUnit::Day),
                other => Err(range_err(&format!("invalid largestUnit: {other}"))),
            };
        }
    }
    Ok(def)
}

pub fn format_iso_date(year: i64, month: i64, day: i64) -> String {
    // tc39 TemporalDateToString / ISO year padding: years outside 0..=9999
    // use a sign + 6 digits.
    if (0..=9999).contains(&year) {
        format!("{:04}-{:02}-{:02}", year, month, day)
    } else {
        let sign = if year < 0 { '-' } else { '+' };
        format!("{}{:06}-{:02}-{:02}", sign, year.abs(), month, day)
    }
}

fn attach_plain_time_methods(o: &Rc<RefCell<HashMap<String, Value>>>) {
    {
        let (h, mi, s, ms, us, ns) = pt_fields(o);
        let mut b = o.borrow_mut();
        b.insert("hour".into(), num(h));
        b.insert("minute".into(), num(mi));
        b.insert("second".into(), num(s));
        b.insert("millisecond".into(), num(ms));
        b.insert("microsecond".into(), num(us));
        b.insert("nanosecond".into(), num(ns));
    }
    meth!(o, "toString", |_i, _a| {
        let o = brand_this("PlainTime")?;
        let (h, mi, s, ms, us, ns) = pt_fields(&o);
        Ok(Value::str(format_iso_time(h, mi, s, ms, us, ns)))
    });
    meth!(o, "toJSON", |_i, _a| {
        let o = brand_this("PlainTime")?;
        let (h, mi, s, ms, us, ns) = pt_fields(&o);
        Ok(Value::str(format_iso_time(h, mi, s, ms, us, ns)))
    });
    meth!(o, "add", |_i, a: Vec<Value>| {
        let o = brand_this("PlainTime")?;
        plain_time_add(&o, &a, 1)
    });
    meth!(o, "subtract", |_i, a: Vec<Value>| {
        let o = brand_this("PlainTime")?;
        plain_time_add(&o, &a, -1)
    });
    meth!(o, "with", |_i, a: Vec<Value>| {
        let o = brand_this("PlainTime")?;
        let (h, mi, s, ms, us, ns) = pt_fields(&o);
        let arg = a.first().cloned().unwrap_or(Value::Undefined);
        let ov = read_overflow(&a, 1)?;
        let nh = arg_field(&arg, "hour", h);
        let nm = arg_field(&arg, "minute", mi);
        let nss = arg_field(&arg, "second", s);
        let nms = arg_field(&arg, "millisecond", ms);
        let nus = arg_field(&arg, "microsecond", us);
        let nns = arg_field(&arg, "nanosecond", ns);
        match regulate_iso_time(nh, nm, nss, nms, nus, nns, ov) {
            Some((a, b, c, d, e, f)) => Ok(mk_plain_time(a, b, c, d, e, f)),
            None => Err(range_err("PlainTime.with: time out of range")),
        }
    });
    meth!(o, "equals", |_i, a: Vec<Value>| {
        let o = brand_this("PlainTime")?;
        let me = time_to_nanos_fields(&o);
        let other = other_plain_time(&a)?;
        Ok(Value::Bool(me == other))
    });
}

fn time_to_nanos_fields(o: &Rc<RefCell<HashMap<String, Value>>>) -> i64 {
    let (h, mi, s, ms, us, ns) = pt_fields(o);
    time_to_nanos(h, mi, s, ms, us, ns)
}

fn other_plain_time(a: &[Value]) -> Result<i64, JsError> {
    match a.first() {
        Some(Value::Object(o))
            if matches!(o.borrow().get("_temporalKind"), Some(Value::String(k)) if &**k == "PlainTime") =>
        {
            Ok(time_to_nanos_fields(o))
        }
        _ => Err(type_err("expected a Temporal.PlainTime")),
    }
}

fn plain_time_add(
    o: &Rc<RefCell<HashMap<String, Value>>>,
    a: &[Value],
    sign: i64,
) -> Result<Value, JsError> {
    let dur = a.first().cloned().unwrap_or(Value::Undefined);
    let (_, _, _, _, hh, mm, ss, ms, us, ns) = read_full_duration(&dur)?;
    let base = time_to_nanos_fields(o) as i128;
    let delta = (sign as i128)
        * (((((hh as i128 * 60 + mm as i128) * 60 + ss as i128) * 1000 + ms as i128) * 1000
            + us as i128)
            * 1000
            + ns as i128);
    let (_days, h, mi, s, msv, usv, nsv) = balance_time(base + delta);
    Ok(mk_plain_time(h, mi, s, msv, usv, nsv))
}

fn pdt_date_fields(o: &Rc<RefCell<HashMap<String, Value>>>) -> (i64, i64, i64) {
    (
        obj_num(o, "isoYear"),
        obj_num(o, "isoMonth"),
        obj_num(o, "isoDay"),
    )
}

fn attach_plain_date_time_methods(o: &Rc<RefCell<HashMap<String, Value>>>) {
    {
        let (y, m, d) = pdt_date_fields(o);
        let (h, mi, s, ms, us, ns) = pt_fields(o);
        let mut b = o.borrow_mut();
        b.insert("year".into(), num(y));
        b.insert("month".into(), num(m));
        b.insert("day".into(), num(d));
        b.insert("hour".into(), num(h));
        b.insert("minute".into(), num(mi));
        b.insert("second".into(), num(s));
        b.insert("millisecond".into(), num(ms));
        b.insert("microsecond".into(), num(us));
        b.insert("nanosecond".into(), num(ns));
        b.insert("dayOfWeek".into(), num(day_of_week(y, m, d)));
        b.insert("inLeapYear".into(), Value::Bool(is_iso_leap_year(y)));
    }
    meth!(o, "toString", |_i, _a| {
        let o = brand_this("PlainDateTime")?;
        let (y, m, d) = pdt_date_fields(&o);
        let (h, mi, s, ms, us, ns) = pt_fields(&o);
        Ok(Value::str(format!(
            "{}T{}",
            format_iso_date(y, m, d),
            format_iso_time(h, mi, s, ms, us, ns)
        )))
    });
    meth!(o, "toJSON", |_i, _a| {
        let o = brand_this("PlainDateTime")?;
        let (y, m, d) = pdt_date_fields(&o);
        let (h, mi, s, ms, us, ns) = pt_fields(&o);
        Ok(Value::str(format!(
            "{}T{}",
            format_iso_date(y, m, d),
            format_iso_time(h, mi, s, ms, us, ns)
        )))
    });
    meth!(o, "toPlainDate", |_i, _a| {
        let o = brand_this("PlainDateTime")?;
        let (y, m, d) = pdt_date_fields(&o);
        Ok(mk_plain_date(y, m, d))
    });
    meth!(o, "toPlainTime", |_i, _a| {
        let o = brand_this("PlainDateTime")?;
        let (h, mi, s, ms, us, ns) = pt_fields(&o);
        Ok(mk_plain_time(h, mi, s, ms, us, ns))
    });
    meth!(o, "add", |_i, a: Vec<Value>| {
        let o = brand_this("PlainDateTime")?;
        plain_date_time_add(&o, &a, 1)
    });
    meth!(o, "subtract", |_i, a: Vec<Value>| {
        let o = brand_this("PlainDateTime")?;
        plain_date_time_add(&o, &a, -1)
    });
    meth!(o, "with", |_i, a: Vec<Value>| {
        let o = brand_this("PlainDateTime")?;
        let (y, m, d) = pdt_date_fields(&o);
        let (h, mi, s, ms, us, ns) = pt_fields(&o);
        let arg = a.first().cloned().unwrap_or(Value::Undefined);
        let ov = read_overflow(&a, 1)?;
        let ny = arg_field(&arg, "year", y);
        let nm = arg_field(&arg, "month", m);
        let nd = arg_field(&arg, "day", d);
        let nh = arg_field(&arg, "hour", h);
        let nmi = arg_field(&arg, "minute", mi);
        let nss = arg_field(&arg, "second", s);
        let nms = arg_field(&arg, "millisecond", ms);
        let nus = arg_field(&arg, "microsecond", us);
        let nns = arg_field(&arg, "nanosecond", ns);
        let (ry, rm, rd) = regulate_iso_date(ny, nm, nd, ov)
            .ok_or_else(|| range_err("PlainDateTime.with: date out of range"))?;
        let (rh, rmi, rs, rms, rus, rns) = regulate_iso_time(nh, nmi, nss, nms, nus, nns, ov)
            .ok_or_else(|| range_err("PlainDateTime.with: time out of range"))?;
        Ok(mk_plain_date_time(ry, rm, rd, rh, rmi, rs, rms, rus, rns))
    });
    meth!(o, "equals", |_i, a: Vec<Value>| {
        let o = brand_this("PlainDateTime")?;
        let (ay, am, ad) = pdt_date_fields(&o);
        let at = time_to_nanos_fields(&o);
        let other = match a.first() {
            Some(Value::Object(b))
                if matches!(b.borrow().get("_temporalKind"), Some(Value::String(k)) if &**k == "PlainDateTime") =>
            {
                let (by, bm, bd) = pdt_date_fields(b);
                let bt = time_to_nanos_fields(b);
                (by, bm, bd, bt)
            }
            _ => return Err(type_err("expected a Temporal.PlainDateTime")),
        };
        Ok(Value::Bool(
            (ay, am, ad, at) == (other.0, other.1, other.2, other.3),
        ))
    });
}

fn plain_date_time_add(
    o: &Rc<RefCell<HashMap<String, Value>>>,
    a: &[Value],
    sign: i64,
) -> Result<Value, JsError> {
    let (y, m, d) = pdt_date_fields(o);
    let (h, mi, s, ms, us, ns) = pt_fields(o);
    let dur = a.first().cloned().unwrap_or(Value::Undefined);
    let ov = read_overflow(a, 1)?;
    let (years, months, weeks, days, hh, mm, ss, dms, dus, dns) = read_full_duration(&dur)?;
    // tc39 AddDateTime: add time first → day overflow carries into the date.
    let base_ns = time_to_nanos(h, mi, s, ms, us, ns) as i128;
    let delta_ns = (sign as i128)
        * (((((hh as i128 * 60 + mm as i128) * 60 + ss as i128) * 1000 + dms as i128) * 1000
            + dus as i128)
            * 1000
            + dns as i128);
    let (time_days, nh, nmi, ns2, nms, nus, nns) = balance_time(base_ns + delta_ns);
    // Then add the date duration + carried time-days.
    let (ny, nm, nd) = add_iso_date(
        y,
        m,
        d,
        sign * years,
        sign * months,
        sign * weeks,
        sign * days + time_days,
        ov,
    )
    .ok_or_else(|| range_err("PlainDateTime.add: result out of range"))?;
    Ok(mk_plain_date_time(ny, nm, nd, nh, nmi, ns2, nms, nus, nns))
}

fn attach_instant_methods(o: &Rc<RefCell<HashMap<String, Value>>>) {
    {
        let ns = read_instant_ns(o);
        let mut b = o.borrow_mut();
        b.insert(
            "epochMilliseconds".into(),
            Value::Number((ns.div_euclid(1_000_000)) as f64),
        );
        // epochNanoseconds is a BigInt in spec; we expose the decimal string
        // under epochNanoseconds (BigInt not always available cross-tier).
        b.insert(
            "epochNanoseconds".into(),
            Value::str(ns.to_string()),
        );
    }
    meth!(o, "toString", |_i, _a| {
        let o = brand_this("Instant")?;
        Ok(Value::str(format_instant(read_instant_ns(&o))))
    });
    meth!(o, "toJSON", |_i, _a| {
        let o = brand_this("Instant")?;
        Ok(Value::str(format_instant(read_instant_ns(&o))))
    });
    meth!(o, "add", |_i, a: Vec<Value>| {
        let o = brand_this("Instant")?;
        instant_add(&o, &a, 1)
    });
    meth!(o, "subtract", |_i, a: Vec<Value>| {
        let o = brand_this("Instant")?;
        instant_add(&o, &a, -1)
    });
    meth!(o, "until", |_i, a: Vec<Value>| {
        let o = brand_this("Instant")?;
        let me = read_instant_ns(&o);
        let other = other_instant(&a)?;
        let diff = other - me; // ns
        Ok(duration_from_total_ns(diff))
    });
    meth!(o, "since", |_i, a: Vec<Value>| {
        let o = brand_this("Instant")?;
        let me = read_instant_ns(&o);
        let other = other_instant(&a)?;
        let diff = me - other; // ns
        Ok(duration_from_total_ns(diff))
    });
    meth!(o, "equals", |_i, a: Vec<Value>| {
        let o = brand_this("Instant")?;
        Ok(Value::Bool(read_instant_ns(&o) == other_instant(&a)?))
    });
}

/// Build a Duration carrying only nanoseconds (Instant arithmetic keeps the
/// exact ns count without rounding; balancing to larger units would lose
/// exactness, so until/since on Instant default to nanoseconds — matching the
/// spec's default `largestUnit: 'second'` produces seconds+sub-second, but a
/// pure-ns Duration is exact and lossless; we balance into h/m/s/ms/us/ns).
fn duration_from_total_ns(total_ns: i128) -> Value {
    let sign = if total_ns < 0 { -1i64 } else { 1 };
    let mut r = total_ns.unsigned_abs();
    let ns = (r % 1000) as i64;
    r /= 1000;
    let us = (r % 1000) as i64;
    r /= 1000;
    let ms = (r % 1000) as i64;
    r /= 1000;
    let s = (r % 60) as i64;
    r /= 60;
    let mi = (r % 60) as i64;
    r /= 60;
    let h = r as i64;
    mk_duration(
        0,
        0,
        0,
        0,
        sign * h,
        sign * mi,
        sign * s,
        sign * ms,
        sign * us,
        sign * ns,
    )
}

fn instant_add(
    o: &Rc<RefCell<HashMap<String, Value>>>,
    a: &[Value],
    sign: i64,
) -> Result<Value, JsError> {
    let dur = a.first().cloned().unwrap_or(Value::Undefined);
    let (years, months, weeks, days, hh, mm, ss, ms, us, ns) = read_full_duration(&dur)?;
    if years != 0 || months != 0 || weeks != 0 {
        return Err(range_err(
            "Instant arithmetic does not support calendar units (years/months/weeks)",
        ));
    }
    let delta_ns: i128 = (sign as i128)
        * ((((((days as i128 * 24 + hh as i128) * 60 + mm as i128) * 60 + ss as i128) * 1000
            + ms as i128)
            * 1000
            + us as i128)
            * 1000
            + ns as i128);
    Ok(mk_instant(read_instant_ns(o) + delta_ns))
}

fn other_instant(a: &[Value]) -> Result<i128, JsError> {
    match a.first() {
        Some(Value::Object(o))
            if matches!(o.borrow().get("_temporalKind"), Some(Value::String(k)) if &**k == "Instant") =>
        {
            Ok(read_instant_ns(o))
        }
        Some(Value::String(s)) => {
            parse_instant_str(s).ok_or_else(|| range_err("invalid Instant string"))
        }
        _ => Err(type_err("expected a Temporal.Instant")),
    }
}

/// Format an Instant (epoch ns) as `YYYY-MM-DDTHH:MM:SS[.fff…]Z`.
pub fn format_instant(epoch_ns: i128) -> String {
    let days = epoch_ns.div_euclid(NS_PER_DAY_I128) as i64;
    let nanos_of_day = epoch_ns.rem_euclid(NS_PER_DAY_I128);
    let (y, m, d) = civil_from_epoch_days(days);
    let (_carry, h, mi, s, ms, us, ns) = balance_time(nanos_of_day);
    format!(
        "{}T{}Z",
        format_iso_date(y, m, d),
        format_iso_time(h, mi, s, ms, us, ns)
    )
}

fn attach_duration_methods(o: &Rc<RefCell<HashMap<String, Value>>>) {
    {
        let s = duration_sign(o);
        let mut b = o.borrow_mut();
        b.insert("sign".into(), num(s));
        b.insert("blank".into(), Value::Bool(s == 0));
    }
    meth!(o, "toString", |_i, _a| {
        let o = brand_this("Duration")?;
        Ok(Value::str(format_duration(&o)))
    });
    meth!(o, "toJSON", |_i, _a| {
        let o = brand_this("Duration")?;
        Ok(Value::str(format_duration(&o)))
    });
    meth!(o, "negated", |_i, _a| {
        let o = brand_this("Duration")?;
        Ok(duration_map(&o, |v| -v))
    });
    meth!(o, "abs", |_i, _a| {
        let o = brand_this("Duration")?;
        Ok(duration_map(&o, |v| v.abs()))
    });
    meth!(o, "with", |_i, a: Vec<Value>| {
        let o = brand_this("Duration")?;
        let arg = a.first().cloned().unwrap_or(Value::Undefined);
        if !matches!(arg, Value::Object(_)) {
            return Err(type_err("Duration.with requires a fields object"));
        }
        Ok(mk_duration(
            arg_field(&arg, "years", obj_num(&o, "years")),
            arg_field(&arg, "months", obj_num(&o, "months")),
            arg_field(&arg, "weeks", obj_num(&o, "weeks")),
            arg_field(&arg, "days", obj_num(&o, "days")),
            arg_field(&arg, "hours", obj_num(&o, "hours")),
            arg_field(&arg, "minutes", obj_num(&o, "minutes")),
            arg_field(&arg, "seconds", obj_num(&o, "seconds")),
            arg_field(&arg, "milliseconds", obj_num(&o, "milliseconds")),
            arg_field(&arg, "microseconds", obj_num(&o, "microseconds")),
            arg_field(&arg, "nanoseconds", obj_num(&o, "nanoseconds")),
        ))
    });
    meth!(o, "add", |_i, a: Vec<Value>| {
        let o = brand_this("Duration")?;
        duration_add(&o, &a, 1)
    });
    meth!(o, "subtract", |_i, a: Vec<Value>| {
        let o = brand_this("Duration")?;
        duration_add(&o, &a, -1)
    });
    meth!(o, "total", |_i, a: Vec<Value>| {
        let o = brand_this("Duration")?;
        duration_total(&o, &a)
    });
}

const DUR_KEYS: [&str; 10] = [
    "years",
    "months",
    "weeks",
    "days",
    "hours",
    "minutes",
    "seconds",
    "milliseconds",
    "microseconds",
    "nanoseconds",
];

fn duration_sign(o: &Rc<RefCell<HashMap<String, Value>>>) -> i64 {
    for k in DUR_KEYS {
        let v = obj_num(o, k);
        if v != 0 {
            return if v > 0 { 1 } else { -1 };
        }
    }
    0
}

fn duration_map<F: Fn(i64) -> i64>(o: &Rc<RefCell<HashMap<String, Value>>>, f: F) -> Value {
    mk_duration(
        f(obj_num(o, "years")),
        f(obj_num(o, "months")),
        f(obj_num(o, "weeks")),
        f(obj_num(o, "days")),
        f(obj_num(o, "hours")),
        f(obj_num(o, "minutes")),
        f(obj_num(o, "seconds")),
        f(obj_num(o, "milliseconds")),
        f(obj_num(o, "microseconds")),
        f(obj_num(o, "nanoseconds")),
    )
}

fn duration_add(
    o: &Rc<RefCell<HashMap<String, Value>>>,
    a: &[Value],
    sign: i64,
) -> Result<Value, JsError> {
    let other = a.first().cloned().unwrap_or(Value::Undefined);
    let (oy, omo, ow, od, oh, omi, os, oms, ous, ons) = read_full_duration(&other)?;
    Ok(mk_duration(
        obj_num(o, "years") + sign * oy,
        obj_num(o, "months") + sign * omo,
        obj_num(o, "weeks") + sign * ow,
        obj_num(o, "days") + sign * od,
        obj_num(o, "hours") + sign * oh,
        obj_num(o, "minutes") + sign * omi,
        obj_num(o, "seconds") + sign * os,
        obj_num(o, "milliseconds") + sign * oms,
        obj_num(o, "microseconds") + sign * ous,
        obj_num(o, "nanoseconds") + sign * ons,
    ))
}

/// Duration.total({unit}) for *time* units. Calendar units (years/months/
/// weeks) require a relativeTo reference (followup); requesting one throws a
/// RangeError per the spec rather than fabricating a result.
fn duration_total(
    o: &Rc<RefCell<HashMap<String, Value>>>,
    a: &[Value],
) -> Result<Value, JsError> {
    let unit = match a.first() {
        Some(Value::String(s)) => (**s).to_string(),
        Some(Value::Object(opt)) => match opt.borrow().get("unit") {
            Some(Value::String(s)) => (**s).to_string(),
            _ => return Err(range_err("Duration.total requires a unit")),
        },
        _ => return Err(range_err("Duration.total requires a unit")),
    };
    if obj_num(o, "years") != 0 || obj_num(o, "months") != 0 || obj_num(o, "weeks") != 0 {
        return Err(range_err(
            "Duration.total with calendar units requires relativeTo (not yet supported)",
        ));
    }
    // Total nanoseconds from days..nanoseconds.
    let total_ns: i128 = (obj_num(o, "days") as i128 * 24 * 3600
        + obj_num(o, "hours") as i128 * 3600
        + obj_num(o, "minutes") as i128 * 60
        + obj_num(o, "seconds") as i128)
        * 1_000_000_000
        + obj_num(o, "milliseconds") as i128 * 1_000_000
        + obj_num(o, "microseconds") as i128 * 1000
        + obj_num(o, "nanoseconds") as i128;
    let divisor: i128 = match unit.as_str() {
        "day" | "days" => NS_PER_DAY_I128,
        "hour" | "hours" => 3_600_000_000_000,
        "minute" | "minutes" => 60_000_000_000,
        "second" | "seconds" => 1_000_000_000,
        "millisecond" | "milliseconds" => 1_000_000,
        "microsecond" | "microseconds" => 1000,
        "nanosecond" | "nanoseconds" => 1,
        other => return Err(range_err(&format!("invalid unit for total(): {other}"))),
    };
    Ok(Value::Number(total_ns as f64 / divisor as f64))
}

/// ISO 8601 duration serialization (tc39 TemporalDurationToString). e.g.
/// `P1Y2M3DT4H5M6S`. A zero duration is `PT0S`.
pub fn format_duration(o: &Rc<RefCell<HashMap<String, Value>>>) -> String {
    let sign = duration_sign(o);
    let g = |k: &str| obj_num(o, k).abs();
    let (y, mo, w, d) = (g("years"), g("months"), g("weeks"), g("days"));
    let (h, mi, s) = (g("hours"), g("minutes"), g("seconds"));
    let (ms, us, ns) = (g("milliseconds"), g("microseconds"), g("nanoseconds"));
    let mut out = String::new();
    if sign < 0 {
        out.push('-');
    }
    out.push('P');
    if y != 0 {
        out.push_str(&format!("{y}Y"));
    }
    if mo != 0 {
        out.push_str(&format!("{mo}M"));
    }
    if w != 0 {
        out.push_str(&format!("{w}W"));
    }
    if d != 0 {
        out.push_str(&format!("{d}D"));
    }
    // Fractional seconds: combine s + ms/us/ns into a decimal.
    let frac_ns = ms * 1_000_000 + us * 1000 + ns;
    let has_time = h != 0 || mi != 0 || s != 0 || frac_ns != 0;
    if has_time {
        out.push('T');
        if h != 0 {
            out.push_str(&format!("{h}H"));
        }
        if mi != 0 {
            out.push_str(&format!("{mi}M"));
        }
        if s != 0 || frac_ns != 0 {
            if frac_ns != 0 {
                let mut digits = format!("{:09}", frac_ns);
                while digits.ends_with('0') {
                    digits.pop();
                }
                out.push_str(&format!("{s}.{digits}S"));
            } else {
                out.push_str(&format!("{s}S"));
            }
        }
    }
    if out == "P" || out == "-P" {
        out.push_str("T0S");
    }
    out
}

fn attach_zoned_methods(o: &Rc<RefCell<HashMap<String, Value>>>) {
    {
        // Read all needed fields BEFORE taking the mutable borrow (obj_str
        // borrows immutably — calling it while `b` is held panics).
        let ns = read_instant_ns(o);
        let tz = obj_str(o, "timeZone");
        let mut b = o.borrow_mut();
        b.insert(
            "epochMilliseconds".into(),
            Value::Number((ns.div_euclid(1_000_000)) as f64),
        );
        b.insert("epochNanoseconds".into(), Value::str(ns.to_string()));
        if let Some(tz) = tz {
            b.insert("timeZoneId".into(), Value::str(tz));
        }
    }
    meth!(o, "toInstant", |_i, _a| {
        let o = brand_this("ZonedDateTime")?;
        Ok(mk_instant(read_instant_ns(&o)))
    });
    meth!(o, "toPlainDateTime", |_i, _a| {
        let o = brand_this("ZonedDateTime")?;
        let ns = read_instant_ns(&o);
        let off_min = obj_num(&o, "offsetMinutes");
        let local = ns + (off_min as i128) * 60 * 1_000_000_000;
        let days = local.div_euclid(NS_PER_DAY_I128) as i64;
        let nod = local.rem_euclid(NS_PER_DAY_I128);
        let (y, m, d) = civil_from_epoch_days(days);
        let (_c, h, mi, s, ms, us, nsx) = balance_time(nod);
        Ok(mk_plain_date_time(y, m, d, h, mi, s, ms, us, nsx))
    });
    meth!(o, "toString", |_i, _a| {
        let o = brand_this("ZonedDateTime")?;
        let ns = read_instant_ns(&o);
        let off_min = obj_num(&o, "offsetMinutes");
        let tz = obj_str(&o, "timeZone").unwrap_or_else(|| "UTC".into());
        let local = ns + (off_min as i128) * 60 * 1_000_000_000;
        let days = local.div_euclid(NS_PER_DAY_I128) as i64;
        let nod = local.rem_euclid(NS_PER_DAY_I128);
        let (y, m, d) = civil_from_epoch_days(days);
        let (_c, h, mi, s, ms, us, nsx) = balance_time(nod);
        let off_str = format_offset(off_min);
        Ok(Value::str(format!(
            "{}T{}{}[{}]",
            format_iso_date(y, m, d),
            format_iso_time(h, mi, s, ms, us, nsx),
            off_str,
            tz
        )))
    });
    meth!(o, "add", |_i, a: Vec<Value>| {
        let o = brand_this("ZonedDateTime")?;
        zoned_add(&o, &a, 1)
    });
    meth!(o, "subtract", |_i, a: Vec<Value>| {
        let o = brand_this("ZonedDateTime")?;
        zoned_add(&o, &a, -1)
    });
    meth!(o, "equals", |_i, a: Vec<Value>| {
        let o = brand_this("ZonedDateTime")?;
        let other = match a.first() {
            Some(Value::Object(b))
                if matches!(b.borrow().get("_temporalKind"), Some(Value::String(k)) if &**k == "ZonedDateTime") =>
            {
                (read_instant_ns(b), obj_str(b, "timeZone"))
            }
            _ => return Err(type_err("expected a Temporal.ZonedDateTime")),
        };
        Ok(Value::Bool(
            read_instant_ns(&o) == other.0 && obj_str(&o, "timeZone") == other.1,
        ))
    });
}

fn format_offset(off_min: i64) -> String {
    let sign = if off_min < 0 { '-' } else { '+' };
    let a = off_min.abs();
    format!("{}{:02}:{:02}", sign, a / 60, a % 60)
}

fn zoned_add(
    o: &Rc<RefCell<HashMap<String, Value>>>,
    a: &[Value],
    sign: i64,
) -> Result<Value, JsError> {
    // UTC-offset core: time + day units are exact on the instant; calendar
    // units (years/months/weeks/days) are applied to the local wall-clock
    // date then re-projected through the fixed offset.
    let dur = a.first().cloned().unwrap_or(Value::Undefined);
    let (years, months, weeks, days, hh, mm, ss, ms, us, ns) = read_full_duration(&dur)?;
    let off_min = obj_num(o, "offsetMinutes");
    let tz = obj_str(o, "timeZone").unwrap_or_else(|| "UTC".into());
    let mut local = read_instant_ns(o) + (off_min as i128) * 60 * 1_000_000_000;
    if years != 0 || months != 0 || weeks != 0 || days != 0 {
        let dys = local.div_euclid(NS_PER_DAY_I128) as i64;
        let nod = local.rem_euclid(NS_PER_DAY_I128);
        let (y, m, d) = civil_from_epoch_days(dys);
        let (ny, nm, nd) = add_iso_date(
            y,
            m,
            d,
            sign * years,
            sign * months,
            sign * weeks,
            sign * days,
            Overflow::Constrain,
        )
        .ok_or_else(|| range_err("ZonedDateTime.add: out of range"))?;
        local = epoch_days(ny, nm, nd) as i128 * NS_PER_DAY_I128 + nod;
    }
    let time_ns: i128 = (sign as i128)
        * (((((hh as i128 * 60 + mm as i128) * 60 + ss as i128) * 1000 + ms as i128) * 1000
            + us as i128)
            * 1000
            + ns as i128);
    local += time_ns;
    let new_instant = local - (off_min as i128) * 60 * 1_000_000_000;
    Ok(mk_zoned(new_instant, &tz, off_min))
}

// ============================================================
// ISO-8601 parsing
// ============================================================

/// Parse `YYYY-MM-DD` (+ optional time/offset suffix ignored) → (y, m, d).
pub fn parse_iso_date_str(s: &str) -> Option<(i64, i64, i64)> {
    let s = s.trim();
    let date_part = s.split(['T', ' ']).next().unwrap_or(s);
    // Optional sign prefix for extended (>4-digit) years, e.g. "+010000-01-01".
    let (sign, body) = if let Some(b) = date_part.strip_prefix('-') {
        (-1i64, b)
    } else if let Some(b) = date_part.strip_prefix('+') {
        (1i64, b)
    } else {
        (1i64, date_part)
    };
    let mut parts = body.splitn(3, '-');
    let y: i64 = sign * parts.next()?.parse::<i64>().ok()?;
    let m: i64 = parts.next()?.parse().ok()?;
    let d: i64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    if !(1..=12).contains(&m) || d < 1 || d > iso_days_in_month(y, m) {
        return None;
    }
    Some((y, m, d))
}

/// Parse a full datetime → (y, m, d, h, mi, s, ms, us, ns, offset_min_opt).
type ParsedDT = (i64, i64, i64, i64, i64, i64, i64, i64, i64, Option<i64>);
pub fn parse_iso_datetime_str(s: &str) -> Option<ParsedDT> {
    let s = s.trim();
    // Strip trailing [Zone] annotation.
    let s = if let Some(pos) = s.find('[') {
        &s[..pos]
    } else {
        s
    };
    let (date_part, rest) = match s.split_once(['T', ' ']) {
        Some((d, r)) => (d, Some(r)),
        None => (s, None),
    };
    let (y, m, d) = parse_iso_date_str(date_part)?;
    let mut h = 0;
    let mut mi = 0;
    let mut sec = 0;
    let mut ms = 0;
    let mut us = 0;
    let mut ns = 0;
    let mut offset = None;
    if let Some(rest) = rest {
        // Split off offset/Z.
        let (time, off) = split_offset(rest);
        offset = off;
        // time = HH[:MM[:SS[.fff…]]]
        let mut tp = time.splitn(3, ':');
        h = tp.next().unwrap_or("0").parse().ok()?;
        if let Some(mm) = tp.next() {
            mi = mm.parse().ok()?;
        }
        if let Some(ss) = tp.next() {
            let (whole, frac) = match ss.split_once('.') {
                Some((w, f)) => (w, Some(f)),
                None => (ss, None),
            };
            sec = whole.parse().ok()?;
            if let Some(f) = frac {
                // pad/truncate to 9 digits
                let mut digits = f.to_string();
                while digits.len() < 9 {
                    digits.push('0');
                }
                digits.truncate(9);
                let nanos: i64 = digits.parse().ok()?;
                ms = nanos / 1_000_000;
                us = (nanos / 1000) % 1000;
                ns = nanos % 1000;
            }
        }
    }
    Some((y, m, d, h, mi, sec, ms, us, ns, offset))
}

fn split_offset(time: &str) -> (&str, Option<i64>) {
    if let Some(stripped) = time.strip_suffix('Z').or_else(|| time.strip_suffix('z')) {
        return (stripped, Some(0));
    }
    // Find a +/- after position 0 (so a leading sign isn't mistaken).
    // Scan from a reasonable point (after HH:MM:SS).
    for (i, c) in time.char_indices() {
        if i > 0 && (c == '+' || c == '-') {
            let (t, o) = time.split_at(i);
            return (t, parse_offset_minutes(o));
        }
    }
    (time, None)
}

/// Parse an Instant ISO string (must have Z or numeric offset) → epoch ns.
pub fn parse_instant_str(s: &str) -> Option<i128> {
    let (y, m, d, h, mi, sec, ms, us, ns, off) = parse_iso_datetime_str(s)?;
    let off = off?; // Instant REQUIRES an offset/Z (tc39 ParseTemporalInstant)
    let local_days = epoch_days(y, m, d);
    let local_ns = local_days as i128 * NS_PER_DAY_I128
        + time_to_nanos(h, mi, sec, ms, us, ns) as i128;
    Some(local_ns - (off as i128) * 60 * 1_000_000_000)
}

/// Parse an ISO 8601 duration `P[n]Y[n]M[n]W[n]DT[n]H[n]M[n[.fff]]S`.
/// tc39 ParseTemporalDurationString. Returns the 10 components (fractional
/// seconds split into ms/us/ns). A leading `-`/`+` sets the overall sign.
pub fn parse_iso_duration(s: &str) -> Option<Dur10> {
    let s = s.trim();
    let (sign, body) = if let Some(b) = s.strip_prefix('-') {
        (-1i64, b)
    } else if let Some(b) = s.strip_prefix('+') {
        (1i64, b)
    } else {
        (1i64, s)
    };
    let body = body.strip_prefix(['P', 'p'])?;
    let (date_part, time_part) = match body.split_once(['T', 't']) {
        Some((d, t)) => (d, Some(t)),
        None => (body, None),
    };
    let mut years = 0i64;
    let mut months = 0i64;
    let mut weeks = 0i64;
    let mut days = 0i64;
    // Date designators.
    let mut num = String::new();
    let mut saw_any = false;
    for c in date_part.chars() {
        if c.is_ascii_digit() {
            num.push(c);
        } else {
            let v: i64 = num.parse().ok()?;
            num.clear();
            saw_any = true;
            match c {
                'Y' | 'y' => years = v,
                'M' | 'm' => months = v,
                'W' | 'w' => weeks = v,
                'D' | 'd' => days = v,
                _ => return None,
            }
        }
    }
    if !num.is_empty() {
        return None; // trailing digits without designator
    }
    let mut hours = 0i64;
    let mut minutes = 0i64;
    let mut seconds = 0i64;
    let mut millis = 0i64;
    let mut micros = 0i64;
    let mut nanos = 0i64;
    if let Some(tp) = time_part {
        let mut numt = String::new();
        let mut frac: Option<String> = None;
        for c in tp.chars() {
            if c.is_ascii_digit() {
                if let Some(f) = frac.as_mut() {
                    f.push(c);
                } else {
                    numt.push(c);
                }
            } else if c == '.' || c == ',' {
                frac = Some(String::new());
            } else {
                let v: i64 = numt.parse().ok()?;
                numt.clear();
                saw_any = true;
                match c {
                    'H' | 'h' => hours = v,
                    'M' | 'm' => minutes = v,
                    'S' | 's' => {
                        seconds = v;
                        if let Some(f) = frac.take() {
                            let mut digits = f;
                            while digits.len() < 9 {
                                digits.push('0');
                            }
                            digits.truncate(9);
                            let n: i64 = digits.parse().ok()?;
                            millis = n / 1_000_000;
                            micros = (n / 1000) % 1000;
                            nanos = n % 1000;
                        }
                    }
                    _ => return None,
                }
            }
        }
        if !numt.is_empty() {
            return None;
        }
    }
    if !saw_any {
        // "P" alone or "PT" alone is invalid.
        return None;
    }
    Some((
        sign * years,
        sign * months,
        sign * weeks,
        sign * days,
        sign * hours,
        sign * minutes,
        sign * seconds,
        sign * millis,
        sign * micros,
        sign * nanos,
    ))
}

// ============================================================
// Global installation (constructors + Temporal.Now + static methods)
// ============================================================

/// Install the `Temporal` global with PlainDate/PlainTime/PlainDateTime/
/// Instant/Duration/ZonedDateTime constructors + `Temporal.Now`.
pub fn install(interp: &Interp) {
    let mut temporal: HashMap<String, Value> = HashMap::new();

    temporal.insert("PlainDate".into(), build_plain_date_ctor());
    temporal.insert("PlainTime".into(), build_plain_time_ctor());
    temporal.insert("PlainDateTime".into(), build_plain_date_time_ctor());
    temporal.insert("Instant".into(), build_instant_ctor());
    temporal.insert("Duration".into(), build_duration_ctor());
    temporal.insert("ZonedDateTime".into(), build_zoned_ctor());
    temporal.insert("Now".into(), build_now());

    interp.define_global("Temporal", Value::Object(Rc::new(RefCell::new(temporal))));
}

fn now_epoch_ns() -> i128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as i128)
        .unwrap_or(0)
}

fn build_now() -> Value {
    let mut m: HashMap<String, Value> = HashMap::new();
    m.insert(
        "instant".into(),
        native_fn_with_interp("instant", |_i, _a| Ok(mk_instant(now_epoch_ns()))),
    );
    m.insert(
        "timeZoneId".into(),
        native_fn_with_interp("timeZoneId", |_i, _a| Ok(Value::str("UTC".to_string()))),
    );
    m.insert(
        "epochMilliseconds".into(),
        native_fn_with_interp("epochMilliseconds", |_i, _a| {
            Ok(Value::Number((now_epoch_ns().div_euclid(1_000_000)) as f64))
        }),
    );
    m.insert(
        "plainDateISO".into(),
        native_fn_with_interp("plainDateISO", |_i, _a| {
            let days = now_epoch_ns().div_euclid(NS_PER_DAY_I128) as i64;
            let (y, m, d) = civil_from_epoch_days(days);
            Ok(mk_plain_date(y, m, d))
        }),
    );
    m.insert(
        "plainTimeISO".into(),
        native_fn_with_interp("plainTimeISO", |_i, _a| {
            let nod = now_epoch_ns().rem_euclid(NS_PER_DAY_I128);
            let (_c, h, mi, s, ms, us, ns) = balance_time(nod);
            Ok(mk_plain_time(h, mi, s, ms, us, ns))
        }),
    );
    m.insert(
        "plainDateTimeISO".into(),
        native_fn_with_interp("plainDateTimeISO", |_i, _a| {
            let ns = now_epoch_ns();
            let days = ns.div_euclid(NS_PER_DAY_I128) as i64;
            let nod = ns.rem_euclid(NS_PER_DAY_I128);
            let (y, m, d) = civil_from_epoch_days(days);
            let (_c, h, mi, s, ms, us, nsx) = balance_time(nod);
            Ok(mk_plain_date_time(y, m, d, h, mi, s, ms, us, nsx))
        }),
    );
    m.insert(
        "zonedDateTimeISO".into(),
        native_fn_with_interp("zonedDateTimeISO", |_i, _a| {
            Ok(mk_zoned(now_epoch_ns(), "UTC", 0))
        }),
    );
    Value::Object(Rc::new(RefCell::new(m)))
}

fn ctor_obj(name: &str, ctor: Value, statics: Vec<(&str, Value)>) -> Value {
    let mut m: HashMap<String, Value> = HashMap::new();
    m.insert("_construct".into(), ctor);
    m.insert("name".into(), Value::str(name.to_string()));
    for (k, v) in statics {
        m.insert(k.into(), v);
    }
    Value::Object(Rc::new(RefCell::new(m)))
}

fn build_plain_date_ctor() -> Value {
    let ctor = native_fn_with_interp("PlainDate", |_i, a: Vec<Value>| {
        let y = a.first().map(|v| v.to_number()).unwrap_or(f64::NAN);
        let m = a.get(1).map(|v| v.to_number()).unwrap_or(f64::NAN);
        let d = a.get(2).map(|v| v.to_number()).unwrap_or(f64::NAN);
        if !y.is_finite() || !m.is_finite() || !d.is_finite() {
            return Err(range_err("PlainDate: arguments must be finite"));
        }
        let (y, m, d) = (y as i64, m as i64, d as i64);
        // The constructor uses REJECT semantics (tc39 §3.1.1: throws if the
        // ISO date is not valid — Feb 30, month 13, etc.).
        match regulate_iso_date(y, m, d, Overflow::Reject) {
            Some((ry, rm, rd)) => Ok(mk_plain_date(ry, rm, rd)),
            None => Err(range_err(&format!(
                "PlainDate: {y:04}-{m:02}-{d:02} is not a valid ISO date"
            ))),
        }
    });
    let from = native_fn_with_interp("from", |_i, a: Vec<Value>| {
        let arg = a.first().cloned().unwrap_or(Value::Undefined);
        let ov = read_overflow(&a, 1)?;
        match &arg {
            Value::String(s) => {
                let (y, m, d) = parse_iso_date_str(s)
                    .ok_or_else(|| range_err("PlainDate.from: invalid ISO date string"))?;
                Ok(mk_plain_date(y, m, d))
            }
            Value::Object(o)
                if matches!(o.borrow().get("_temporalKind"), Some(Value::String(k)) if &**k == "PlainDate") =>
            {
                let (y, m, d) = pd_fields(o);
                Ok(mk_plain_date(y, m, d))
            }
            Value::Object(_) => {
                let y = arg_field(&arg, "year", i64::MIN);
                let m = arg_field(&arg, "month", i64::MIN);
                let d = arg_field(&arg, "day", i64::MIN);
                if y == i64::MIN || m == i64::MIN || d == i64::MIN {
                    return Err(type_err("PlainDate.from: missing year/month/day"));
                }
                regulate_iso_date(y, m, d, ov)
                    .map(|(y, m, d)| mk_plain_date(y, m, d))
                    .ok_or_else(|| range_err("PlainDate.from: date out of range"))
            }
            _ => Err(type_err("PlainDate.from: invalid argument")),
        }
    });
    let compare = native_fn_with_interp("compare", |_i, a: Vec<Value>| {
        let (ay, am, ad) = coerce_pd(a.first())?;
        let (by, bm, bd) = coerce_pd(a.get(1))?;
        Ok(num(compare_iso_date(ay, am, ad, by, bm, bd)))
    });
    ctor_obj("PlainDate", ctor, vec![("from", from), ("compare", compare)])
}

fn coerce_pd(v: Option<&Value>) -> Result<(i64, i64, i64), JsError> {
    match v {
        Some(Value::Object(o))
            if matches!(o.borrow().get("_temporalKind"), Some(Value::String(k)) if &**k == "PlainDate") =>
        {
            Ok(pd_fields(o))
        }
        Some(Value::String(s)) => {
            parse_iso_date_str(s).ok_or_else(|| range_err("invalid PlainDate string"))
        }
        _ => Err(type_err("expected a Temporal.PlainDate")),
    }
}

fn build_plain_time_ctor() -> Value {
    let ctor = native_fn_with_interp("PlainTime", |_i, a: Vec<Value>| {
        let g = |i: usize| a.get(i).map(|v| v.to_number()).unwrap_or(0.0);
        let vals = [g(0), g(1), g(2), g(3), g(4), g(5)];
        if vals.iter().any(|v| !v.is_finite()) {
            return Err(range_err("PlainTime: arguments must be finite"));
        }
        let v: Vec<i64> = vals.iter().map(|x| *x as i64).collect();
        match regulate_iso_time(v[0], v[1], v[2], v[3], v[4], v[5], Overflow::Reject) {
            Some((h, mi, s, ms, us, ns)) => Ok(mk_plain_time(h, mi, s, ms, us, ns)),
            None => Err(range_err("PlainTime: time out of range")),
        }
    });
    let from = native_fn_with_interp("from", |_i, a: Vec<Value>| {
        let arg = a.first().cloned().unwrap_or(Value::Undefined);
        match &arg {
            Value::String(s) => {
                let (_, _, _, h, mi, sec, ms, us, ns, _) = parse_iso_datetime_str(s)
                    .or_else(|| {
                        // bare time "HH:MM:SS"
                        parse_iso_datetime_str(&format!("1970-01-01T{s}"))
                    })
                    .ok_or_else(|| range_err("PlainTime.from: invalid string"))?;
                Ok(mk_plain_time(h, mi, sec, ms, us, ns))
            }
            Value::Object(o)
                if matches!(o.borrow().get("_temporalKind"), Some(Value::String(k)) if &**k == "PlainTime") =>
            {
                let (h, mi, s, ms, us, ns) = pt_fields(o);
                Ok(mk_plain_time(h, mi, s, ms, us, ns))
            }
            Value::Object(_) => {
                let ov = read_overflow(&a, 1)?;
                let h = arg_field(&arg, "hour", 0);
                let mi = arg_field(&arg, "minute", 0);
                let s = arg_field(&arg, "second", 0);
                let ms = arg_field(&arg, "millisecond", 0);
                let us = arg_field(&arg, "microsecond", 0);
                let ns = arg_field(&arg, "nanosecond", 0);
                regulate_iso_time(h, mi, s, ms, us, ns, ov)
                    .map(|(a, b, c, d, e, f)| mk_plain_time(a, b, c, d, e, f))
                    .ok_or_else(|| range_err("PlainTime.from: out of range"))
            }
            _ => Err(type_err("PlainTime.from: invalid argument")),
        }
    });
    let compare = native_fn_with_interp("compare", |_i, a: Vec<Value>| {
        let one = coerce_pt(a.first())?;
        let two = coerce_pt(a.get(1))?;
        Ok(num(match one.cmp(&two) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }))
    });
    ctor_obj("PlainTime", ctor, vec![("from", from), ("compare", compare)])
}

fn coerce_pt(v: Option<&Value>) -> Result<i64, JsError> {
    match v {
        Some(Value::Object(o))
            if matches!(o.borrow().get("_temporalKind"), Some(Value::String(k)) if &**k == "PlainTime") =>
        {
            Ok(time_to_nanos_fields(o))
        }
        _ => Err(type_err("expected a Temporal.PlainTime")),
    }
}

fn build_plain_date_time_ctor() -> Value {
    let ctor = native_fn_with_interp("PlainDateTime", |_i, a: Vec<Value>| {
        let gy = a.first().map(|v| v.to_number()).unwrap_or(f64::NAN);
        let gm = a.get(1).map(|v| v.to_number()).unwrap_or(f64::NAN);
        let gd = a.get(2).map(|v| v.to_number()).unwrap_or(f64::NAN);
        if !gy.is_finite() || !gm.is_finite() || !gd.is_finite() {
            return Err(range_err("PlainDateTime: y/m/d must be finite"));
        }
        let gt = |i: usize| a.get(i).map(|v| v.to_number()).unwrap_or(0.0) as i64;
        let (y, m, d) = (gy as i64, gm as i64, gd as i64);
        let (ry, rm, rd) = regulate_iso_date(y, m, d, Overflow::Reject)
            .ok_or_else(|| range_err("PlainDateTime: invalid ISO date"))?;
        let (h, mi, s, ms, us, ns) = (gt(3), gt(4), gt(5), gt(6), gt(7), gt(8));
        let (rh, rmi, rs, rms, rus, rns) =
            regulate_iso_time(h, mi, s, ms, us, ns, Overflow::Reject)
                .ok_or_else(|| range_err("PlainDateTime: invalid ISO time"))?;
        Ok(mk_plain_date_time(ry, rm, rd, rh, rmi, rs, rms, rus, rns))
    });
    let from = native_fn_with_interp("from", |_i, a: Vec<Value>| {
        let arg = a.first().cloned().unwrap_or(Value::Undefined);
        let ov = read_overflow(&a, 1)?;
        match &arg {
            Value::String(s) => {
                let (y, m, d, h, mi, sec, ms, us, ns, _off) = parse_iso_datetime_str(s)
                    .ok_or_else(|| range_err("PlainDateTime.from: invalid string"))?;
                Ok(mk_plain_date_time(y, m, d, h, mi, sec, ms, us, ns))
            }
            Value::Object(o)
                if matches!(o.borrow().get("_temporalKind"), Some(Value::String(k)) if &**k == "PlainDateTime") =>
            {
                let (y, m, d) = pdt_date_fields(o);
                let (h, mi, s, ms, us, ns) = pt_fields(o);
                Ok(mk_plain_date_time(y, m, d, h, mi, s, ms, us, ns))
            }
            Value::Object(_) => {
                let y = arg_field(&arg, "year", i64::MIN);
                let m = arg_field(&arg, "month", i64::MIN);
                let d = arg_field(&arg, "day", i64::MIN);
                if y == i64::MIN || m == i64::MIN || d == i64::MIN {
                    return Err(type_err("PlainDateTime.from: missing year/month/day"));
                }
                let (ry, rm, rd) = regulate_iso_date(y, m, d, ov)
                    .ok_or_else(|| range_err("PlainDateTime.from: date out of range"))?;
                let h = arg_field(&arg, "hour", 0);
                let mi = arg_field(&arg, "minute", 0);
                let s = arg_field(&arg, "second", 0);
                let ms = arg_field(&arg, "millisecond", 0);
                let us = arg_field(&arg, "microsecond", 0);
                let ns = arg_field(&arg, "nanosecond", 0);
                let (rh, rmi, rs, rms, rus, rns) = regulate_iso_time(h, mi, s, ms, us, ns, ov)
                    .ok_or_else(|| range_err("PlainDateTime.from: time out of range"))?;
                Ok(mk_plain_date_time(ry, rm, rd, rh, rmi, rs, rms, rus, rns))
            }
            _ => Err(type_err("PlainDateTime.from: invalid argument")),
        }
    });
    let compare = native_fn_with_interp("compare", |_i, a: Vec<Value>| {
        let one = coerce_pdt(a.first())?;
        let two = coerce_pdt(a.get(1))?;
        Ok(num(match one.cmp(&two) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }))
    });
    ctor_obj(
        "PlainDateTime",
        ctor,
        vec![("from", from), ("compare", compare)],
    )
}

fn coerce_pdt(v: Option<&Value>) -> Result<(i64, i64, i64, i64), JsError> {
    match v {
        Some(Value::Object(o))
            if matches!(o.borrow().get("_temporalKind"), Some(Value::String(k)) if &**k == "PlainDateTime") =>
        {
            let (y, m, d) = pdt_date_fields(o);
            Ok((y, m, d, time_to_nanos_fields(o)))
        }
        _ => Err(type_err("expected a Temporal.PlainDateTime")),
    }
}

fn build_instant_ctor() -> Value {
    let ctor = native_fn_with_interp("Instant", |_i, a: Vec<Value>| {
        // new Temporal.Instant(epochNanoseconds) — accepts a BigInt or a
        // numeric/decimal string (BigInt may be re-homed cross-tier).
        let ns = match a.first() {
            Some(Value::BigInt(b)) => b
                .to_string()
                .parse::<i128>()
                .map_err(|_| range_err("Instant: epochNanoseconds out of range"))?,
            Some(Value::String(s)) => s
                .parse::<i128>()
                .map_err(|_| range_err("Instant: invalid epochNanoseconds string"))?,
            Some(Value::Number(n)) => *n as i128,
            _ => return Err(type_err("Instant: epochNanoseconds required")),
        };
        Ok(mk_instant(ns))
    });
    let from = native_fn_with_interp("from", |_i, a: Vec<Value>| match a.first() {
        Some(Value::String(s)) => {
            parse_instant_str(s).map(mk_instant).ok_or_else(|| range_err("Instant.from: invalid string"))
        }
        Some(Value::Object(o))
            if matches!(o.borrow().get("_temporalKind"), Some(Value::String(k)) if &**k == "Instant") =>
        {
            Ok(mk_instant(read_instant_ns(o)))
        }
        _ => Err(type_err("Instant.from: invalid argument")),
    });
    let from_ms = native_fn_with_interp("fromEpochMilliseconds", |_i, a: Vec<Value>| {
        let ms = a.first().map(|v| v.to_number()).unwrap_or(f64::NAN);
        if !ms.is_finite() {
            return Err(range_err("fromEpochMilliseconds: non-finite"));
        }
        Ok(mk_instant((ms as i128) * 1_000_000))
    });
    let compare = native_fn_with_interp("compare", |_i, a: Vec<Value>| {
        let one = other_instant(std::slice::from_ref(
            a.first().unwrap_or(&Value::Undefined),
        ))?;
        let two = other_instant(std::slice::from_ref(
            a.get(1).unwrap_or(&Value::Undefined),
        ))?;
        Ok(num(match one.cmp(&two) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }))
    });
    ctor_obj(
        "Instant",
        ctor,
        vec![
            ("from", from),
            ("fromEpochMilliseconds", from_ms),
            ("compare", compare),
        ],
    )
}

fn build_duration_ctor() -> Value {
    let ctor = native_fn_with_interp("Duration", |_i, a: Vec<Value>| {
        let g = |i: usize| a.get(i).map(|v| v.to_number()).unwrap_or(0.0);
        let vals: Vec<i64> = (0..10).map(|i| g(i) as i64).collect();
        // tc39 §7.1.1: signs of all nonzero components must agree.
        let mut sign = 0i64;
        for &v in &vals {
            if v != 0 {
                let s = if v > 0 { 1 } else { -1 };
                if sign == 0 {
                    sign = s;
                } else if sign != s {
                    return Err(range_err("Duration: mixed-sign components are not allowed"));
                }
            }
        }
        Ok(mk_duration(
            vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], vals[8],
            vals[9],
        ))
    });
    let from = native_fn_with_interp("from", |_i, a: Vec<Value>| {
        let arg = a.first().cloned().unwrap_or(Value::Undefined);
        match &arg {
            Value::String(s) => {
                let d = parse_iso_duration(s)
                    .ok_or_else(|| range_err("Duration.from: invalid ISO duration"))?;
                Ok(mk_duration(
                    d.0, d.1, d.2, d.3, d.4, d.5, d.6, d.7, d.8, d.9,
                ))
            }
            Value::Object(o)
                if matches!(o.borrow().get("_temporalKind"), Some(Value::String(k)) if &**k == "Duration") =>
            {
                Ok(duration_map(o, |v| v))
            }
            Value::Object(_) => Ok(mk_duration(
                arg_field(&arg, "years", 0),
                arg_field(&arg, "months", 0),
                arg_field(&arg, "weeks", 0),
                arg_field(&arg, "days", 0),
                arg_field(&arg, "hours", 0),
                arg_field(&arg, "minutes", 0),
                arg_field(&arg, "seconds", 0),
                arg_field(&arg, "milliseconds", 0),
                arg_field(&arg, "microseconds", 0),
                arg_field(&arg, "nanoseconds", 0),
            )),
            _ => Err(type_err("Duration.from: invalid argument")),
        }
    });
    let compare = native_fn_with_interp("compare", |_i, a: Vec<Value>| {
        // Compare total nanoseconds for time-only durations (calendar units
        // need relativeTo — followup; throw rather than fake).
        let total = |v: Option<&Value>| -> Result<i128, JsError> {
            let val = v.cloned().unwrap_or(Value::Undefined);
            let (y, mo, w, d, h, mi, s, ms, us, ns) = read_full_duration(&val)?;
            if y != 0 || mo != 0 || w != 0 {
                return Err(range_err(
                    "Duration.compare with calendar units requires relativeTo (followup)",
                ));
            }
            Ok((((((d as i128 * 24 + h as i128) * 60 + mi as i128) * 60 + s as i128) * 1000
                + ms as i128)
                * 1000
                + us as i128)
                * 1000
                + ns as i128)
        };
        let one = total(a.first())?;
        let two = total(a.get(1))?;
        Ok(num(match one.cmp(&two) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }))
    });
    ctor_obj("Duration", ctor, vec![("from", from), ("compare", compare)])
}

fn build_zoned_ctor() -> Value {
    let ctor = native_fn_with_interp("ZonedDateTime", |_i, a: Vec<Value>| {
        let ns = match a.first() {
            Some(Value::BigInt(b)) => b
                .to_string()
                .parse::<i128>()
                .map_err(|_| range_err("ZonedDateTime: epochNanoseconds out of range"))?,
            Some(Value::String(s)) => s
                .parse::<i128>()
                .map_err(|_| range_err("ZonedDateTime: invalid epochNanoseconds"))?,
            Some(Value::Number(n)) => *n as i128,
            _ => return Err(type_err("ZonedDateTime: epochNanoseconds required")),
        };
        let tz = match a.get(1) {
            Some(Value::String(s)) => (**s).to_string(),
            Some(Value::Undefined) | None => {
                return Err(type_err("ZonedDateTime: timeZone required"))
            }
            Some(other) => other.to_display_string(),
        };
        let off = parse_offset_minutes(&tz).unwrap_or(0);
        Ok(mk_zoned(ns, &tz, off))
    });
    let from = native_fn_with_interp("from", |_i, a: Vec<Value>| {
        let arg = a.first().cloned().unwrap_or(Value::Undefined);
        match &arg {
            Value::String(s) => {
                // Parse "YYYY-MM-DDTHH:MM:SS±HH:MM[Zone]"
                let zone = extract_zone(s);
                let parsed = parse_iso_datetime_str(s)
                    .ok_or_else(|| range_err("ZonedDateTime.from: invalid string"))?;
                let off = parsed.9.or_else(|| zone.as_deref().and_then(parse_offset_minutes));
                let off = off.unwrap_or(0);
                let (y, m, d, h, mi, sec, ms, us, ns, _) = parsed;
                let local = epoch_days(y, m, d) as i128 * NS_PER_DAY_I128
                    + time_to_nanos(h, mi, sec, ms, us, ns) as i128;
                let instant = local - (off as i128) * 60 * 1_000_000_000;
                let tz = zone.unwrap_or_else(|| format_offset(off));
                Ok(mk_zoned(instant, &tz, off))
            }
            Value::Object(o)
                if matches!(o.borrow().get("_temporalKind"), Some(Value::String(k)) if &**k == "ZonedDateTime") =>
            {
                let ns = read_instant_ns(o);
                let tz = obj_str(o, "timeZone").unwrap_or_else(|| "UTC".into());
                Ok(mk_zoned(ns, &tz, obj_num(o, "offsetMinutes")))
            }
            _ => Err(type_err("ZonedDateTime.from: invalid argument")),
        }
    });
    let compare = native_fn_with_interp("compare", |_i, a: Vec<Value>| {
        let one = coerce_zdt(a.first())?;
        let two = coerce_zdt(a.get(1))?;
        Ok(num(match one.cmp(&two) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }))
    });
    ctor_obj(
        "ZonedDateTime",
        ctor,
        vec![("from", from), ("compare", compare)],
    )
}

fn coerce_zdt(v: Option<&Value>) -> Result<i128, JsError> {
    match v {
        Some(Value::Object(o))
            if matches!(o.borrow().get("_temporalKind"), Some(Value::String(k)) if &**k == "ZonedDateTime") =>
        {
            Ok(read_instant_ns(o))
        }
        _ => Err(type_err("expected a Temporal.ZonedDateTime")),
    }
}

fn extract_zone(s: &str) -> Option<String> {
    let start = s.find('[')?;
    let end = s.find(']')?;
    if end > start + 1 {
        Some(s[start + 1..end].to_string())
    } else {
        None
    }
}

// ============================================================
// Unit tests — REAL behavior (leap years, overflow, boundary crossing,
// calendar difference, ns arithmetic, ISO formatting/parsing).
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp::Interp;

    /// Run JS through a fully-installed interp (which includes `Temporal` via
    /// `install_basic_globals`) and return the `console.log` lines.
    fn js(src: &str) -> Vec<String> {
        let mut i = Interp::new();
        i.install_basic_globals();
        i.run(src).expect("run");
        i.output
    }

    // ---- End-to-end JS tests: prove the global wiring + spec behavior ----

    #[test]
    fn js_plaindate_construct_and_tostring() {
        assert_eq!(
            js("console.log(new Temporal.PlainDate(2024, 2, 29).toString());"),
            vec!["2024-02-29"]
        );
        // year/month/day getters.
        assert_eq!(
            js("var d = new Temporal.PlainDate(2024, 7, 4); console.log(d.year, d.month, d.day);"),
            vec!["2024 7 4"]
        );
    }

    #[test]
    fn js_plaindate_invalid_leap_throws() {
        // 2023-02-29 is invalid; the constructor uses REJECT -> RangeError.
        assert_eq!(
            js("try { new Temporal.PlainDate(2023, 2, 29); console.log('no'); } \
                catch (e) { console.log(e.name); }"),
            vec!["RangeError"]
        );
        // 2024-02-29 is valid (leap year).
        assert_eq!(
            js("console.log(new Temporal.PlainDate(2024, 2, 29).day);"),
            vec!["29"]
        );
    }

    #[test]
    fn js_plaindate_add_crosses_boundaries_and_is_immutable() {
        // add 1 day crosses month; original is unchanged (immutable).
        assert_eq!(
            js("var d = new Temporal.PlainDate(2024, 1, 31); \
                var d2 = d.add({ days: 1 }); \
                console.log(d2.toString(), d.toString());"),
            vec!["2024-02-01 2024-01-31"]
        );
        // add 1 month constrains Jan 31 -> Feb 29 (leap).
        assert_eq!(
            js("console.log(new Temporal.PlainDate(2024, 1, 31).add({ months: 1 }).toString());"),
            vec!["2024-02-29"]
        );
        // year boundary.
        assert_eq!(
            js("console.log(new Temporal.PlainDate(2024, 12, 31).add({ days: 1 }).toString());"),
            vec!["2025-01-01"]
        );
        // subtract.
        assert_eq!(
            js("console.log(new Temporal.PlainDate(2024, 3, 1).subtract({ days: 1 }).toString());"),
            vec!["2024-02-29"]
        );
    }

    #[test]
    fn js_plaindate_add_reject_throws() {
        assert_eq!(
            js("try { new Temporal.PlainDate(2023, 1, 31).add({ months: 1 }, { overflow: 'reject' }); \
                console.log('no'); } catch (e) { console.log(e.name); }"),
            vec!["RangeError"]
        );
    }

    #[test]
    fn js_plaindate_until_gives_duration() {
        // Calendar difference (default largestUnit auto -> day).
        assert_eq!(
            js("var a = new Temporal.PlainDate(2024, 1, 1); \
                var b = new Temporal.PlainDate(2024, 3, 1); \
                console.log(a.until(b).days);"),
            vec!["60"]
        );
        // largestUnit year.
        assert_eq!(
            js("var a = new Temporal.PlainDate(2020, 2, 29); \
                var b = new Temporal.PlainDate(2023, 3, 1); \
                var dur = a.until(b, { largestUnit: 'year' }); \
                console.log(dur.years, dur.months, dur.days);"),
            vec!["3 0 1"]
        );
        // since is the reverse sign.
        assert_eq!(
            js("var a = new Temporal.PlainDate(2024, 1, 1); \
                var b = new Temporal.PlainDate(2024, 3, 1); \
                console.log(b.since(a).days);"),
            vec!["60"]
        );
    }

    #[test]
    fn js_plaindate_compare_orders() {
        assert_eq!(
            js("var a = new Temporal.PlainDate(2024, 1, 1); \
                var b = new Temporal.PlainDate(2024, 1, 2); \
                console.log(Temporal.PlainDate.compare(a, b), \
                            Temporal.PlainDate.compare(b, a), \
                            Temporal.PlainDate.compare(a, a));"),
            vec!["-1 1 0"]
        );
    }

    #[test]
    fn js_plaindate_from_and_with() {
        assert_eq!(
            js("console.log(Temporal.PlainDate.from('2024-02-29').toString());"),
            vec!["2024-02-29"]
        );
        // with() overrides a field, immutable.
        assert_eq!(
            js("var d = Temporal.PlainDate.from('2024-02-29'); \
                console.log(d.with({ year: 2023 }).toString(), d.toString());"),
            vec!["2023-02-28 2024-02-29"]
        );
    }

    #[test]
    fn js_instant_ns_arithmetic() {
        // Instant from epoch ns, add seconds, toString ISO.
        assert_eq!(
            js("var i = Temporal.Instant.from('2024-01-01T00:00:00Z'); \
                console.log(i.add({ seconds: 90 }).toString());"),
            vec!["2024-01-01T00:01:30Z"]
        );
        // exact nanosecond bookkeeping (no f64 rounding): epochNanoseconds is a
        // decimal string carrying full i128 precision.
        assert_eq!(
            js("var i = new Temporal.Instant('1700000000123456789'); \
                console.log(i.epochNanoseconds);"),
            vec!["1700000000123456789"]
        );
        // until in ns -> Duration of seconds.
        assert_eq!(
            js("var a = Temporal.Instant.from('2024-01-01T00:00:00Z'); \
                var b = Temporal.Instant.from('2024-01-01T00:00:10Z'); \
                console.log(a.until(b).seconds);"),
            vec!["10"]
        );
        // compare.
        assert_eq!(
            js("var a = Temporal.Instant.from('2024-01-01T00:00:00Z'); \
                var b = Temporal.Instant.from('2024-01-01T00:00:01Z'); \
                console.log(Temporal.Instant.compare(a, b));"),
            vec!["-1"]
        );
    }

    #[test]
    fn js_duration_total_and_string() {
        // total() converts time units.
        assert_eq!(
            js("console.log(Temporal.Duration.from({ hours: 1, minutes: 30 }).total({ unit: 'minutes' }));"),
            vec!["90"]
        );
        // toString ISO 8601.
        assert_eq!(
            js("console.log(new Temporal.Duration(1, 2, 0, 3, 4, 5, 6).toString());"),
            vec!["P1Y2M3DT4H5M6S"]
        );
        // negated.
        assert_eq!(
            js("console.log(Temporal.Duration.from('PT1H').negated().toString());"),
            vec!["-PT1H"]
        );
        // mixed-sign components throw.
        assert_eq!(
            js("try { new Temporal.Duration(1, -1); console.log('no'); } \
                catch (e) { console.log(e.name); }"),
            vec!["RangeError"]
        );
    }

    #[test]
    fn js_plaindatetime_time_carries_into_date() {
        // 2024-02-28T23:00 + 25h: +1h -> Feb 29 00:00 (leap), +24h -> Mar 1
        // 00:00. Time overflow carries into the date across the leap day.
        assert_eq!(
            js("var dt = new Temporal.PlainDateTime(2024, 2, 28, 23, 0, 0); \
                console.log(dt.add({ hours: 25 }).toString());"),
            vec!["2024-03-01T00:00:00"]
        );
        // +1h lands exactly on the leap day.
        assert_eq!(
            js("var dt = new Temporal.PlainDateTime(2024, 2, 28, 23, 0, 0); \
                console.log(dt.add({ hours: 1 }).toString());"),
            vec!["2024-02-29T00:00:00"]
        );
    }

    #[test]
    fn js_now_returns_real_values() {
        // Now.instant() is a real Instant whose epoch is > 2020 and parseable.
        assert_eq!(
            js("var i = Temporal.Now.instant(); \
                console.log(typeof i.epochMilliseconds === 'number' && i.epochMilliseconds > 1577836800000);"),
            vec!["true"]
        );
        // plainDateISO() year is sane.
        assert_eq!(
            js("console.log(Temporal.Now.plainDateISO().year >= 2024);"),
            vec!["true"]
        );
    }

    #[test]
    fn js_zoneddatetime_offset_core() {
        // ZonedDateTime from an offset string projects local wall-clock through
        // the fixed offset and back.
        assert_eq!(
            js("var z = Temporal.ZonedDateTime.from('2024-01-01T12:00:00+05:00[+05:00]'); \
                console.log(z.toInstant().toString());"),
            vec!["2024-01-01T07:00:00Z"]
        );
    }

    #[test]
    fn leap_year_rules() {
        assert!(is_iso_leap_year(2024));
        assert!(is_iso_leap_year(2000));
        assert!(!is_iso_leap_year(1900));
        assert!(!is_iso_leap_year(2023));
        assert_eq!(iso_days_in_month(2024, 2), 29);
        assert_eq!(iso_days_in_month(2023, 2), 28);
        assert_eq!(iso_days_in_month(2024, 4), 30);
    }

    #[test]
    fn leap_day_valid_and_invalid() {
        // 2024-02-29 valid; reject keeps it.
        assert_eq!(
            regulate_iso_date(2024, 2, 29, Overflow::Reject),
            Some((2024, 2, 29))
        );
        // 2023-02-29 invalid; reject -> None (constructor throws).
        assert_eq!(regulate_iso_date(2023, 2, 29, Overflow::Reject), None);
        // constrain clamps Feb 29 (non-leap) to Feb 28.
        assert_eq!(
            regulate_iso_date(2023, 2, 31, Overflow::Constrain),
            Some((2023, 2, 28))
        );
        // Feb 31 (leap) -> Feb 29.
        assert_eq!(
            regulate_iso_date(2024, 2, 31, Overflow::Constrain),
            Some((2024, 2, 29))
        );
    }

    #[test]
    fn add_crosses_month_and_year() {
        // 2024-01-31 + 1 day -> 2024-02-01
        assert_eq!(
            add_iso_date(2024, 1, 31, 0, 0, 0, 1, Overflow::Constrain),
            Some((2024, 2, 1))
        );
        // 2024-12-31 + 1 day -> 2025-01-01
        assert_eq!(
            add_iso_date(2024, 12, 31, 0, 0, 0, 1, Overflow::Constrain),
            Some((2025, 1, 1))
        );
        // 2024-01-31 + 1 month -> 2024-02-29 (constrain) [leap]
        assert_eq!(
            add_iso_date(2024, 1, 31, 0, 1, 0, 0, Overflow::Constrain),
            Some((2024, 2, 29))
        );
        // 2023-01-31 + 1 month -> 2023-02-28 (constrain) [non-leap]
        assert_eq!(
            add_iso_date(2023, 1, 31, 0, 1, 0, 0, Overflow::Constrain),
            Some((2023, 2, 28))
        );
        // 2020-01-31 + 13 months -> 2021-02-28 (year+month normalize)
        assert_eq!(
            add_iso_date(2020, 1, 31, 0, 13, 0, 0, Overflow::Constrain),
            Some((2021, 2, 28))
        );
        // subtract via negative days: 2024-03-01 - 1 day -> 2024-02-29
        assert_eq!(
            add_iso_date(2024, 3, 1, 0, 0, 0, -1, Overflow::Constrain),
            Some((2024, 2, 29))
        );
        // add weeks
        assert_eq!(
            add_iso_date(2024, 1, 1, 0, 0, 2, 0, Overflow::Constrain),
            Some((2024, 1, 15))
        );
    }

    #[test]
    fn add_month_reject_overflow() {
        // 2023-01-31 + 1 month with reject -> Feb 31 invalid -> None.
        assert_eq!(
            add_iso_date(2023, 1, 31, 0, 1, 0, 0, Overflow::Reject),
            None
        );
    }

    #[test]
    fn epoch_day_roundtrip_and_known_values() {
        assert_eq!(epoch_days(1970, 1, 1), 0);
        assert_eq!(epoch_days(1969, 12, 31), -1);
        assert_eq!(epoch_days(2000, 1, 1), 10957);
        for &(y, m, d) in &[
            (1, 1, 1),
            (1969, 12, 31),
            (1970, 1, 1),
            (2024, 2, 29),
            (2024, 12, 31),
            (9999, 12, 31),
            (-1, 6, 15),
        ] {
            let ed = epoch_days(y, m, d);
            assert_eq!(civil_from_epoch_days(ed), (y, m, d), "roundtrip {y}-{m}-{d}");
        }
    }

    #[test]
    fn day_of_week_known() {
        // 1970-01-01 was Thursday (ISO 4).
        assert_eq!(day_of_week(1970, 1, 1), 4);
        // 2024-02-29 was a Thursday.
        assert_eq!(day_of_week(2024, 2, 29), 4);
        // 2000-01-01 was a Saturday (ISO 6).
        assert_eq!(day_of_week(2000, 1, 1), 6);
    }

    #[test]
    fn difference_days_and_calendar() {
        // until between two dates, largestUnit day.
        let (_, _, _, days) =
            difference_iso_date(2024, 1, 1, 2024, 3, 1, DateUnit::Day);
        assert_eq!(days, 60); // Jan(31) + Feb(29) = 60 (leap year)
        // backward gives negative.
        let (_, _, _, days_b) =
            difference_iso_date(2024, 3, 1, 2024, 1, 1, DateUnit::Day);
        assert_eq!(days_b, -60);
        // calendar largestUnit year: 2020-02-29 .. 2023-03-01.
        let (y, mo, _, d) =
            difference_iso_date(2020, 2, 29, 2023, 3, 1, DateUnit::Year);
        // 3 years -> 2023-02-28 (constrain), then +1 day -> 2023-03-01.
        assert_eq!((y, mo, d), (3, 0, 1));
        // largestUnit month between 2024-01-15 and 2024-03-20.
        let (_, mm, _, dd) =
            difference_iso_date(2024, 1, 15, 2024, 3, 20, DateUnit::Month);
        assert_eq!((mm, dd), (2, 5));
        // Round-trip: start.add(until) == end.
        let (yy, mo2, _, dd2) =
            difference_iso_date(2019, 11, 30, 2024, 2, 29, DateUnit::Year);
        let back =
            add_iso_date(2019, 11, 30, yy, mo2, 0, dd2, Overflow::Constrain).unwrap();
        assert_eq!(back, (2024, 2, 29));
    }

    #[test]
    fn week_difference() {
        let (_, _, w, d) = difference_iso_date(2024, 1, 1, 2024, 1, 25, DateUnit::Week);
        assert_eq!((w, d), (3, 3)); // 24 days = 3 weeks 3 days
    }

    #[test]
    fn compare_orders_dates() {
        assert_eq!(compare_iso_date(2024, 1, 1, 2024, 1, 2), -1);
        assert_eq!(compare_iso_date(2024, 1, 2, 2024, 1, 1), 1);
        assert_eq!(compare_iso_date(2024, 1, 1, 2024, 1, 1), 0);
        assert_eq!(compare_iso_date(2023, 12, 31, 2024, 1, 1), -1);
    }

    #[test]
    fn time_balance_and_format() {
        // 23:59:59.999999999 + 1ns -> next day 00:00:00.
        let total = time_to_nanos(23, 59, 59, 999, 999, 999) as i128 + 1;
        let (days, h, mi, s, ms, us, ns) = balance_time(total);
        assert_eq!((days, h, mi, s, ms, us, ns), (1, 0, 0, 0, 0, 0, 0));
        // format omits zero fraction.
        assert_eq!(format_iso_time(9, 5, 3, 0, 0, 0), "09:05:03");
        // precision "auto": trim ALL trailing zeros (500ms -> ".5").
        assert_eq!(format_iso_time(1, 2, 3, 500, 0, 0), "01:02:03.5");
        assert_eq!(format_iso_time(1, 2, 3, 0, 0, 1), "01:02:03.000000001");
        assert_eq!(format_iso_time(1, 2, 3, 123, 456, 0), "01:02:03.123456");
    }

    #[test]
    fn iso_date_format() {
        assert_eq!(format_iso_date(2024, 2, 9), "2024-02-09");
        assert_eq!(format_iso_date(98, 1, 1), "0098-01-01");
        assert_eq!(format_iso_date(10000, 1, 1), "+010000-01-01");
        assert_eq!(format_iso_date(-1, 1, 1), "-000001-01-01");
    }

    #[test]
    fn parse_iso_date_works() {
        assert_eq!(parse_iso_date_str("2024-02-29"), Some((2024, 2, 29)));
        assert_eq!(parse_iso_date_str("2024-02-29T12:00:00"), Some((2024, 2, 29)));
        assert_eq!(parse_iso_date_str("2023-02-29"), None); // invalid
        assert_eq!(parse_iso_date_str("2024-13-01"), None); // bad month
    }

    #[test]
    fn parse_instant_works() {
        // 1970-01-01T00:00:00Z == 0 ns.
        assert_eq!(parse_instant_str("1970-01-01T00:00:00Z"), Some(0));
        // 1970-01-01T00:00:01Z == 1e9 ns.
        assert_eq!(parse_instant_str("1970-01-01T00:00:01Z"), Some(1_000_000_000));
        // offset applied: 1970-01-01T01:00:00+01:00 == 0 ns.
        assert_eq!(
            parse_instant_str("1970-01-01T01:00:00+01:00"),
            Some(0)
        );
        // fractional seconds.
        assert_eq!(
            parse_instant_str("1970-01-01T00:00:00.5Z"),
            Some(500_000_000)
        );
        // no offset -> None (Instant requires offset).
        assert_eq!(parse_instant_str("1970-01-01T00:00:00"), None);
    }

    #[test]
    fn instant_format_roundtrip() {
        assert_eq!(format_instant(0), "1970-01-01T00:00:00Z");
        assert_eq!(format_instant(1_000_000_000), "1970-01-01T00:00:01Z");
        assert_eq!(
            format_instant(500_000_000),
            "1970-01-01T00:00:00.5Z"
        );
        // Roundtrip a real timestamp.
        let ns = parse_instant_str("2024-02-29T13:45:30.250Z").unwrap();
        assert_eq!(format_instant(ns), "2024-02-29T13:45:30.25Z");
    }

    #[test]
    fn parse_duration_works() {
        assert_eq!(
            parse_iso_duration("P1Y2M3DT4H5M6S"),
            Some((1, 2, 0, 3, 4, 5, 6, 0, 0, 0))
        );
        assert_eq!(
            parse_iso_duration("P1W"),
            Some((0, 0, 1, 0, 0, 0, 0, 0, 0, 0))
        );
        assert_eq!(
            parse_iso_duration("PT0.5S"),
            Some((0, 0, 0, 0, 0, 0, 0, 500, 0, 0))
        );
        assert_eq!(
            parse_iso_duration("-P1D"),
            Some((0, 0, 0, -1, 0, 0, 0, 0, 0, 0))
        );
        assert_eq!(parse_iso_duration("P"), None);
        assert_eq!(parse_iso_duration("PT"), None);
        assert_eq!(parse_iso_duration("garbage"), None);
    }

    #[test]
    fn offset_parse() {
        assert_eq!(parse_offset_minutes("Z"), Some(0));
        assert_eq!(parse_offset_minutes("+05:30"), Some(330));
        assert_eq!(parse_offset_minutes("-08:00"), Some(-480));
        assert_eq!(parse_offset_minutes("+0530"), Some(330));
        assert_eq!(parse_offset_minutes("+09"), Some(540));
        assert_eq!(parse_offset_minutes("garbage"), None);
    }
}
