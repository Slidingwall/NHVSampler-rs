use std::collections::HashMap;
use std::fs;
use std::path::Path;
#[derive(Debug, Clone)]
enum Node {
    Dict(HashMap<String, Node>),
    #[allow(dead_code)]
    List(Vec<Node>),
    F32(f32),
    #[allow(dead_code)]
    Farr(Vec<f32>),
    Bytes(Vec<u8>),
    NodeArr(Vec<Node>),
}
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}
impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0 }
    }
    fn u8(&mut self) -> Option<u8> {
        let b = *self.buf.get(self.pos)?;
        self.pos += 1;
        Some(b)
    }
    fn i32(&mut self) -> Option<i32> {
        let s = self.buf.get(self.pos..self.pos + 4)?;
        self.pos += 4;
        Some(i32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn f32(&mut self) -> Option<f32> {
        let s = self.buf.get(self.pos..self.pos + 4)?;
        self.pos += 4;
        Some(f32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.buf.get(self.pos..self.pos + n)?;
        self.pos += n;
        Some(s)
    }
    fn seek(&mut self, p: usize) {
        self.pos = p;
    }
}
fn parse_node(c: &mut Cursor) -> Option<Node> {
    let tag = c.u8()?;
    match tag {
        1 => {
            let n = c.i32()? as usize;
            let mut m = HashMap::with_capacity(n);
            for _ in 0..n {
                let kl = c.u8()? as usize;
                let kb = c.take(kl)?;
                let key = String::from_utf8_lossy(kb).into_owned();
                let v = parse_node(c)?;
                m.insert(key, v);
            }
            Some(Node::Dict(m))
        }
        2 => {
            let d1 = c.i32()? as usize;
            let d2 = c.i32()? as usize;
            let mut v = Vec::with_capacity(d1 * d2);
            for _ in 0..d1 * d2 {
                v.push(parse_node(c)?);
            }
            Some(Node::List(v))
        }
        3 => Some(Node::F32(c.f32()?)),
        5 => {
            let d1 = c.i32()? as usize;
            let d2 = c.i32()? as usize;
            let n = d1 * d2;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(c.f32()?);
            }
            Some(Node::Farr(v))
        }
        6 => {
            let n = c.i32()? as usize;
            let b = c.take(n)?.to_vec();
            Some(Node::Bytes(b))
        }
        7 => {
            let cnt = c.i32()? as usize;
            let mut offs = Vec::with_capacity(cnt + 1);
            for _ in 0..cnt + 1 {
                offs.push(c.i32()? as usize);
            }
            let mut items = Vec::with_capacity(cnt);
            for i in 0..cnt {
                c.seek(offs[i]);
                items.push(parse_node(c)?);
            }
            Some(Node::NodeArr(items))
        }
        _ => None,
    }
}
fn as_dict<'a>(n: &'a Node, k: &str) -> Option<&'a Node> {
    if let Node::Dict(m) = n {
        m.get(k)
    } else {
        None
    }
}
fn as_f32(n: &Node) -> Option<f32> {
    if let Node::F32(v) = n {
        Some(*v)
    } else {
        None
    }
}
fn frame_f0(fr: &Node) -> f32 {
    if let Node::Dict(d) = fr {
        if let Some(b40) = d.get("_40") {
            if let Node::Bytes(b) = b40 {
                if b.len() >= 4 {
                    return f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
            }
        }
    }
    0.0
}
pub fn read_llsm_vuv(
    path: &Path,
    n_origin: usize,
    origin_hop: usize,
    sr: u32,
) -> Option<Vec<f32>> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() < 5 || &bytes[0..5] != b"\x04data" {
        return None;
    }
    let mut c = Cursor::new(&bytes);
    c.seek(5);
    let root = parse_node(&mut c)?;
    let frames = match as_dict(&root, "_3a") {
        Some(Node::NodeArr(f)) => f,
        _ => return None,
    };
    if frames.is_empty() {
        return None;
    }
    let mut hop_sec = 0.0f32;
    if let Some(cfg) = as_dict(&root, "_35") {
        if let Some(h) = as_dict(cfg, "_c").and_then(as_f32) {
            if h > 0.0 {
                hop_sec = h;
            }
        }
    }
    if hop_sec <= 0.0 {
        let dur = as_dict(&root, "duration").and_then(as_f32).unwrap_or(0.0);
        if dur > 0.0 {
            hop_sec = dur / frames.len() as f32;
        }
    }
    let hop_samples = if hop_sec > 0.0 {
        (hop_sec * sr as f32).max(1.0)
    } else {
        256.0 
    };
    let n_frames = frames.len() as f32;
    let mut uv = Vec::with_capacity(n_origin);
    for j in 0..n_origin {
        let t = (j * origin_hop) as f32;
        let fi = ((t / hop_samples).floor()).min(n_frames - 1.0) as usize;
        let f0 = frame_f0(&frames[fi]);
        uv.push(if f0 > 0.0 { 0.0 } else { 1.0 });
    }
    Some(uv)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn make_llsm(hop_sec: f32, f0s: &[f32]) -> Vec<u8> {
        let mut root = Vec::new();
        root.push(1u8); 
        root.extend_from_slice(&(4i32).to_le_bytes()); 
        root.push(7u8);
        root.extend_from_slice(b"version");
        root.push(3u8); 
        root.extend_from_slice(&(524.0f32).to_le_bytes());
        root.push(8u8);
        root.extend_from_slice(b"duration");
        root.push(3u8); 
        root.extend_from_slice(&(f0s.len() as f32 * hop_sec).to_le_bytes());
        root.push(3u8);
        root.extend_from_slice(b"_35");
        root.push(1u8); 
        root.extend_from_slice(&(2i32).to_le_bytes());
        root.push(2u8); 
        root.extend_from_slice(b"_c");
        root.push(3u8);
        root.extend_from_slice(&hop_sec.to_le_bytes());
        root.push(2u8); 
        root.extend_from_slice(b"_1");
        root.push(3u8); 
        root.extend_from_slice(&(f0s.len() as f32).to_le_bytes());
        root.push(3u8);
        root.extend_from_slice(b"_3a");
        root.push(7u8);
        root.extend_from_slice(&(f0s.len() as i32).to_le_bytes());
        let off_patch = root.len();
        for _ in 0..f0s.len() + 1 {
            root.extend_from_slice(&[0u8; 4]);
        }
        let mut offs = Vec::new();
        for &f0 in f0s {
            offs.push(root.len() + 5);
            root.push(1u8); 
            root.extend_from_slice(&(1i32).to_le_bytes());
            root.push(3u8);
            root.extend_from_slice(b"_40");
            root.push(6u8); 
            root.extend_from_slice(&(4i32).to_le_bytes());
            root.extend_from_slice(&f0.to_le_bytes());
        }
        offs.push(root.len() + 5);
        for (i, o) in offs.iter().enumerate() {
            let p = off_patch + i * 4;
            root[p..p + 4].copy_from_slice(&(*o as i32).to_le_bytes());
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"\x04data");
        out.extend_from_slice(&root);
        out
    }
    #[test]
    fn read_llsm_vuv_basic() {
        let hop_sec = 256.0 / 44100.0;
        let f0s = [0.0f32, 0.0, 0.0, 220.0, 330.0, 0.0, 440.0, 0.0];
        let bytes = make_llsm(hop_sec, &f0s);
        let dir = std::env::temp_dir();
        let p = dir.join("nhv_llsm_vuv_test.llsm");
        fs::write(&p, &bytes).unwrap();
        let n_origin = 16; 
        let uv = read_llsm_vuv(&p, n_origin, 128, 44100).expect("read");
        assert_eq!(uv.len(), n_origin);
        assert_eq!(uv[0], 1.0);
        assert_eq!(uv[5], 1.0);
        assert_eq!(uv[6], 0.0);
        assert_eq!(uv[7], 0.0);
        fs::remove_file(&p).ok();
    }
    #[test]
    fn read_llsm_vuv_missing_file_is_none() {
        let uv = read_llsm_vuv(
            Path::new("E:/does_not_exist_xyz/__no__.wav.llsm"),
            10,
            128,
            44100,
        );
        assert!(uv.is_none());
    }
}