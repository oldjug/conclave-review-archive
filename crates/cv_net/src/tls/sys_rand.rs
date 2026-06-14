//! OS-backed CSPRNG via `BCryptGenRandom` with `BCRYPT_USE_SYSTEM_PREFERRED_RNG`.

#![allow(non_camel_case_types, non_snake_case)]

type NTSTATUS = i32;
type ULONG = u32;
type PUCHAR = *mut u8;

const BCRYPT_USE_SYSTEM_PREFERRED_RNG: ULONG = 0x0000_0002;

#[link(name = "bcrypt")]
unsafe extern "system" {
    fn BCryptGenRandom(
        hAlgorithm: *mut core::ffi::c_void,
        pbBuffer: PUCHAR,
        cbBuffer: ULONG,
        dwFlags: ULONG,
    ) -> NTSTATUS;
}

pub(crate) fn fill(buf: &mut [u8]) {
    if buf.is_empty() {
        return;
    }
    let rc = unsafe {
        BCryptGenRandom(
            core::ptr::null_mut(),
            buf.as_mut_ptr(),
            buf.len() as ULONG,
            BCRYPT_USE_SYSTEM_PREFERRED_RNG,
        )
    };
    assert!(rc == 0, "BCryptGenRandom failed: 0x{:08x}", rc as u32);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonzero_random() {
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        fill(&mut a);
        fill(&mut b);
        assert_ne!(a, b, "two random draws should differ");
        assert!(a.iter().any(|&v| v != 0));
    }
}
