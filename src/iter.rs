use core::slice;

pub struct Bytes<'a> {
    slice: &'a [u8],
    pos: usize
}

impl<'a> Bytes<'a> {
    #[inline]
    pub fn new(slice: &'a [u8]) -> Bytes<'a> {
        Bytes {
            slice: slice,
            pos: 0
        }
    }

    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    pub fn peek(&self) -> Option<u8> {
        self.slice.get(self.pos).cloned()
    }

    #[inline]
    pub fn bump(&mut self) {
        self.pos += 1;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    #[inline]
    pub fn slice(&mut self) -> &'a [u8] {
        self.slice_skip(0)
    }

    #[inline]
    pub fn slice_skip(&mut self, skip: usize) -> &'a [u8] {
        debug_assert!(self.pos >= skip);
        let head_pos = self.pos - skip;
        unsafe {
            let ptr = self.slice.as_ptr();
            let head = slice::from_raw_parts(ptr, head_pos);
            let tail = slice::from_raw_parts(ptr.offset(self.pos as isize), self.slice.len() - self.pos);
            self.pos = 0;
            self.slice = tail;
            head
        }
    }

    #[inline]
    pub fn next_8<'b>(&'b mut self) -> Option<Bytes8<'b, 'a>> {
        if self.slice.len() > self.pos + 8 {
            Some(Bytes8 { bytes: self, pos: 0 })
        } else {
            None
        }
    }

    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn find(&mut self, _ranges: &[u8]) -> Option<u8> {
        None
    }

    #[cfg(feature = "simd")]
    #[inline]
    pub fn find(&mut self, ranges: &[u8]) -> Option<u8> {
        return find_inner(self, ranges);


        #[cfg(not(target_feature = "sse4.2"))]
        #[inline]
        fn find_inner(_bytes: &mut Bytes, _ranges: &[u8]) -> Option<u8> {
            None
        }

        #[cfg(target_feature = "sse4.2")]
        #[inline]
        fn find_inner(bytes: &mut Bytes, ranges: &[u8]) -> Option<u8> {
            use simd::{i8x16, u8x16};

            extern "platform-intrinsic" {
                fn x86_mm_cmpestri(x: i8x16, y: i32, z: i8x16, w: i32, a: i32) -> i32;
            }
            
            const _SIDD_CMP_RANGES: i32 = 0b0100;

            let ranges_len = ranges.len() as i32;
            let mut blank = [0; 16];
            blank[..ranges.len()].copy_from_slice(ranges);
            let ranges = u8x16::load(&blank, 0).to_i8();

            let mut left = (bytes.slice.len() - bytes.pos) & !15;
            while left != 0 {
                let vec = u8x16::load(&bytes.slice, bytes.pos).to_i8();
                let i = unsafe {
                    x86_mm_cmpestri(ranges, ranges_len, vec, 16, _SIDD_CMP_RANGES)
                };
                if i != 16 {
                    bytes.pos += i as usize + 1;
                    return Some(unsafe { *bytes.slice.get_unchecked(bytes.pos - 1) });
                }
                bytes.pos += 16;
                left -= 16;
            }
            None
        }
    }
}

impl<'a> Iterator for Bytes<'a> {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<u8> {
        if self.slice.len() > self.pos {
            let b = unsafe { *self.slice.get_unchecked(self.pos) };
            self.pos += 1;
            Some(b)
        } else {
            None
        }
    }
}

pub struct Bytes8<'a, 'b: 'a> {
    bytes: &'a mut Bytes<'b>,
    pos: usize
}

macro_rules! bytes8_methods {
    ($f:ident, $pos:expr) => {
        #[inline]
        pub fn $f(&mut self) -> u8 {
            debug_assert!(self.assert_pos($pos));
            let b = unsafe { *self.bytes.slice.get_unchecked(self.bytes.pos) };
            self.bytes.pos += 1;
            b
        }
    };
    () => {
        bytes8_methods!(_0, 0);
        bytes8_methods!(_1, 1);
        bytes8_methods!(_2, 2);
        bytes8_methods!(_3, 3);
        bytes8_methods!(_4, 4);
        bytes8_methods!(_5, 5);
        bytes8_methods!(_6, 6);
        bytes8_methods!(_7, 7);
    }
}

impl<'a, 'b: 'a> Bytes8<'a, 'b> {
    bytes8_methods! {}

    fn assert_pos(&mut self, pos: usize) -> bool {
        let ret = self.pos == pos;
        self.pos += 1;
        ret
    }
}
