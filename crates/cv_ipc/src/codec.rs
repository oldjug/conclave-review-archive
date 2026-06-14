//! Wire-format codecs.
//!
//! Every primitive lays itself down in little-endian. Composite types
//! (String, Vec<T>) prefix a u32 count. Bytes is a Vec<u8> with the
//! same prefix. Encoders write into a growable Vec<u8>; decoders read
//! from a slice with a cursor.

#[derive(Debug, PartialEq, Eq)]
pub enum DecodeError {
    Truncated,
    InvalidUtf8,
    OutOfRange,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => f.write_str("truncated payload"),
            Self::InvalidUtf8 => f.write_str("invalid UTF-8"),
            Self::OutOfRange => f.write_str("out of range"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// A growable buffer used during encoding. Encoders append little-
/// endian bytes; the caller eventually frames them into a message.
#[derive(Debug, Default)]
pub struct Writer {
    pub bytes: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(cap),
        }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn write_u8(&mut self, v: u8) {
        self.bytes.push(v);
    }

    pub fn write_u16(&mut self, v: u16) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    pub fn write_u32(&mut self, v: u32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    pub fn write_u64(&mut self, v: u64) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    pub fn write_i32(&mut self, v: i32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    pub fn write_i64(&mut self, v: i64) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    pub fn write_f32(&mut self, v: f32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    pub fn write_f64(&mut self, v: f64) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    pub fn write_bool(&mut self, v: bool) {
        self.write_u8(if v { 1 } else { 0 });
    }

    pub fn write_bytes(&mut self, b: &[u8]) {
        self.write_u32(b.len() as u32);
        self.bytes.extend_from_slice(b);
    }

    pub fn write_str(&mut self, s: &str) {
        self.write_bytes(s.as_bytes());
    }
}

/// A cursor into an encoded payload. Decoders pull little-endian
/// primitives off the front.
#[derive(Debug)]
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        if self.remaining() < n {
            return Err(DecodeError::Truncated);
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    pub fn read_u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }

    pub fn read_u16(&mut self) -> Result<u16, DecodeError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub fn read_u32(&mut self) -> Result<u32, DecodeError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn read_u64(&mut self) -> Result<u64, DecodeError> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }

    pub fn read_i32(&mut self) -> Result<i32, DecodeError> {
        Ok(self.read_u32()? as i32)
    }

    pub fn read_i64(&mut self) -> Result<i64, DecodeError> {
        Ok(self.read_u64()? as i64)
    }

    pub fn read_f32(&mut self) -> Result<f32, DecodeError> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    pub fn read_f64(&mut self) -> Result<f64, DecodeError> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    pub fn read_bool(&mut self) -> Result<bool, DecodeError> {
        Ok(self.read_u8()? != 0)
    }

    pub fn read_bytes(&mut self) -> Result<Vec<u8>, DecodeError> {
        let len = self.read_u32()? as usize;
        Ok(self.take(len)?.to_vec())
    }

    pub fn read_str(&mut self) -> Result<String, DecodeError> {
        let bytes = self.read_bytes()?;
        String::from_utf8(bytes).map_err(|_| DecodeError::InvalidUtf8)
    }
}

/// Encode-yourself trait. Implemented for the primitives above and any
/// user message struct that needs a wire form.
pub trait Encode {
    fn encode(&self, w: &mut Writer);
}

/// Decode-yourself trait.
pub trait Decode: Sized {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError>;
}

macro_rules! impl_primitive {
    ($t:ty, $write:ident, $read:ident) => {
        impl Encode for $t {
            fn encode(&self, w: &mut Writer) {
                w.$write(*self);
            }
        }
        impl Decode for $t {
            fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
                r.$read()
            }
        }
    };
}

impl_primitive!(u8, write_u8, read_u8);
impl_primitive!(u16, write_u16, read_u16);
impl_primitive!(u32, write_u32, read_u32);
impl_primitive!(u64, write_u64, read_u64);
impl_primitive!(i32, write_i32, read_i32);
impl_primitive!(i64, write_i64, read_i64);
impl_primitive!(f32, write_f32, read_f32);
impl_primitive!(f64, write_f64, read_f64);
impl_primitive!(bool, write_bool, read_bool);

impl Encode for String {
    fn encode(&self, w: &mut Writer) {
        w.write_str(self);
    }
}

impl Decode for String {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        r.read_str()
    }
}

impl<T: Encode> Encode for Vec<T> {
    fn encode(&self, w: &mut Writer) {
        w.write_u32(self.len() as u32);
        for item in self {
            item.encode(w);
        }
    }
}

impl<T: Decode> Decode for Vec<T> {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let n = r.read_u32()? as usize;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(T::decode(r)?);
        }
        Ok(out)
    }
}

impl<T: Encode> Encode for Option<T> {
    fn encode(&self, w: &mut Writer) {
        match self {
            Some(v) => {
                w.write_bool(true);
                v.encode(w);
            }
            None => w.write_bool(false),
        }
    }
}

impl<T: Decode> Decode for Option<T> {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let present = r.read_bool()?;
        if present {
            Ok(Some(T::decode(r)?))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_primitives() {
        let mut w = Writer::new();
        13u32.encode(&mut w);
        (-7i32).encode(&mut w);
        true.encode(&mut w);
        false.encode(&mut w);
        std::f64::consts::PI.encode(&mut w);
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        assert_eq!(u32::decode(&mut r).unwrap(), 13);
        assert_eq!(i32::decode(&mut r).unwrap(), -7);
        assert!(bool::decode(&mut r).unwrap());
        assert!(!bool::decode(&mut r).unwrap());
        let pi = f64::decode(&mut r).unwrap();
        assert!((pi - std::f64::consts::PI).abs() < 1e-12);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn round_trip_string_and_vec() {
        let mut w = Writer::new();
        let s = "hello — Conclave".to_string();
        s.encode(&mut w);
        let v: Vec<u32> = vec![1, 2, 3, 4, 5];
        v.encode(&mut w);
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        assert_eq!(String::decode(&mut r).unwrap(), s);
        assert_eq!(Vec::<u32>::decode(&mut r).unwrap(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn round_trip_option_and_nested_vec() {
        let mut w = Writer::new();
        let some: Option<String> = Some("yes".into());
        let none: Option<String> = None;
        some.encode(&mut w);
        none.encode(&mut w);
        let nested: Vec<Vec<u8>> = vec![vec![1, 2, 3], vec![], vec![9]];
        nested.encode(&mut w);
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        assert_eq!(
            Option::<String>::decode(&mut r).unwrap(),
            Some("yes".into())
        );
        assert_eq!(Option::<String>::decode(&mut r).unwrap(), None);
        assert_eq!(
            Vec::<Vec<u8>>::decode(&mut r).unwrap(),
            vec![vec![1, 2, 3], vec![], vec![9]]
        );
    }

    #[test]
    fn truncated_returns_error() {
        let bytes = vec![0u8, 0, 0]; // u32 needs 4 bytes
        let mut r = Reader::new(&bytes);
        assert_eq!(u32::decode(&mut r), Err(DecodeError::Truncated));
    }
}
