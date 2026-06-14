//! DNS over HTTPS (RFC 8484) — client that POSTs a binary DNS query
//! to a configurable resolver endpoint and parses the response. V1
//! supports A + AAAA record queries via the wire format (not
//! `application/dns-json`). The default resolver is Cloudflare's
//! `1.1.1.1`; the host is hard-coded to `cloudflare-dns.com` so the
//! TLS SNI + cert match the resolver IP.

use crate::dns::IpAddr;
use crate::http1::{Client, Request};
use cv_url::Url;

/// Build a DNS query message per RFC 1035 §4.1 for the given hostname
/// and QTYPE. Returns the wire bytes ready to POST as an opaque body.
fn build_query(name: &str, qtype: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(name.len() + 18);
    // Header: id=0 (RFC 8484 §4.1 recommends always 0 for caching),
    // flags=0x0100 (RD=1), QDCOUNT=1, rest=0.
    out.extend_from_slice(&[0, 0]);
    out.extend_from_slice(&[0x01, 0x00]);
    out.extend_from_slice(&[0, 1]);
    out.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
    // QNAME — length-prefixed labels then terminating 0.
    for label in name.split('.') {
        if label.is_empty() {
            continue;
        }
        let bytes = label.as_bytes();
        if bytes.len() > 63 {
            return Vec::new();
        }
        out.push(bytes.len() as u8);
        out.extend_from_slice(bytes);
    }
    out.push(0);
    // QTYPE + QCLASS=IN.
    out.extend_from_slice(&qtype.to_be_bytes());
    out.extend_from_slice(&1u16.to_be_bytes());
    out
}

/// Parse a DNS response into a list of A/AAAA addresses. Returns an
/// empty list when the answer count is zero or the response is
/// malformed. The pointer-decompression for CNAME chains is handled
/// inline; we skip past CNAMEs to the IP RDATA at the end.
fn parse_answer(msg: &[u8]) -> Vec<IpAddr> {
    if msg.len() < 12 {
        return Vec::new();
    }
    let qd = u16::from_be_bytes([msg[4], msg[5]]) as usize;
    let an = u16::from_be_bytes([msg[6], msg[7]]) as usize;
    let mut i = 12usize;
    fn skip_name(msg: &[u8], mut i: usize) -> usize {
        while i < msg.len() {
            let b = msg[i];
            if b == 0 {
                return i + 1;
            }
            if b & 0xC0 == 0xC0 {
                return i + 2;
            }
            i += 1 + b as usize;
        }
        i
    }
    for _ in 0..qd {
        i = skip_name(msg, i);
        i += 4; // QTYPE + QCLASS
    }
    let mut out = Vec::new();
    for _ in 0..an {
        if i >= msg.len() {
            break;
        }
        i = skip_name(msg, i);
        if i + 10 > msg.len() {
            break;
        }
        let rtype = u16::from_be_bytes([msg[i], msg[i + 1]]);
        let rdlen = u16::from_be_bytes([msg[i + 8], msg[i + 9]]) as usize;
        i += 10;
        if i + rdlen > msg.len() {
            break;
        }
        if rtype == 1 && rdlen == 4 {
            let mut a = [0u8; 4];
            a.copy_from_slice(&msg[i..i + 4]);
            out.push(IpAddr::V4(a));
        } else if rtype == 28 && rdlen == 16 {
            let mut a = [0u8; 16];
            a.copy_from_slice(&msg[i..i + 16]);
            out.push(IpAddr::V6(a));
        }
        i += rdlen;
    }
    out
}

/// Resolve a host via DoH. `endpoint` should be an `https://` URL such
/// as `https://cloudflare-dns.com/dns-query`. Returns A + AAAA IPs
/// pooled together (caller chooses preference). Bytes are returned;
/// connect/handshake errors surface as `Err`.
pub fn resolve(endpoint: &Url, host: &str, client: &Client) -> Result<Vec<IpAddr>, String> {
    let mut out = Vec::new();
    for qtype in [1u16, 28] {
        let body = build_query(host, qtype);
        if body.is_empty() {
            continue;
        }
        let req = Request {
            method: "POST".into(),
            url: endpoint.clone(),
            headers: vec![
                ("Content-Type".into(), "application/dns-message".into()),
                ("Accept".into(), "application/dns-message".into()),
            ],
            body,
            accept_brotli: true,
        };
        let resp = client.send(req).map_err(|e| e.to_string())?;
        if resp.status != 200 {
            continue;
        }
        out.extend(parse_answer(&resp.body));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_starts_with_zero_id_and_rd_flag() {
        let q = build_query("example.com", 1);
        assert_eq!(&q[0..4], &[0, 0, 0x01, 0x00]);
    }

    #[test]
    fn parse_minimal_a_answer() {
        // QD=0, AN=1 with one A 1.2.3.4 record (name compressed pointer 0xC00C).
        let mut msg = vec![
            0, 0, 0x81, 0x80, 0, 0, 0, 1, 0, 0, 0, 0, // header
            0xC0, 0x0C, // name pointer
            0, 1, // type A
            0, 1, // class IN
            0, 0, 0, 60, // TTL
            0, 4, // RDLENGTH
            1, 2, 3, 4,
        ];
        // Fix qd=0 since we skipped question section.
        msg[5] = 0;
        let ips = parse_answer(&msg);
        assert_eq!(ips.len(), 1);
        assert!(matches!(&ips[0], IpAddr::V4([1, 2, 3, 4])));
    }
}
