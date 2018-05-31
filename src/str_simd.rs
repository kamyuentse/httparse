#[cfg(target_arch = "x86")]
use core::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_feature = "avx2")]
#[inline]
pub fn _strspn_ascii_32_avx(buf: &[u8], arbm: __m256i) -> usize {
    debug_assert!(buf.len() >= 32);

    let ptr = buf.as_ptr();

    #[allow(non_snake_case, overflowing_literals)]
    unsafe {
        let LSH: __m256i = _mm256_set1_epi8(0x0f);
        let ARF: __m256i = _mm256_setr_epi8(
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        );

        let dat = _mm256_lddqu_si256(ptr as *const _);
        let rbm = _mm256_shuffle_epi8(arbm, dat);
        let col = _mm256_and_si256(LSH, _mm256_srli_epi16(dat, 4));
        let bit = _mm256_and_si256(_mm256_shuffle_epi8(ARF, col), rbm);
        let msk = _mm256_cmpeq_epi8(bit, _mm256_setzero_si256());

        let res = 0xffff_ffff_0000_0000 | _mm256_movemask_epi8(msk) as u64;

        _tzcnt_u64(res) as usize
    }
}

#[cfg(target_feature = "avx2")]
#[inline]
pub fn _strspn_ascii_64_avx(buf: &[u8], arbm: __m256i) -> usize {
    debug_assert!(buf.len() >= 64);

    let ptr = buf.as_ptr();

    #[allow(non_snake_case, overflowing_literals)]
    unsafe {
        let LSH: __m256i = _mm256_set1_epi8(0x0f);
        let ARF: __m256i = _mm256_setr_epi8(
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        );

        let dat0 = _mm256_lddqu_si256(ptr as *const _);
        let dat1 = _mm256_lddqu_si256(ptr.offset(32) as *const _);

        let rbm0 = _mm256_shuffle_epi8(arbm, dat0);
        let rbm1 = _mm256_shuffle_epi8(arbm, dat1);

        let col0 = _mm256_and_si256(LSH, _mm256_srli_epi16(dat0, 4));
        let col1 = _mm256_and_si256(LSH, _mm256_srli_epi16(dat1, 4));

        let bit0 = _mm256_and_si256(_mm256_shuffle_epi8(ARF, col0), rbm0);
        let bit1 = _mm256_and_si256(_mm256_shuffle_epi8(ARF, col1), rbm1);

        let msk0 = _mm256_cmpeq_epi8(bit0, _mm256_setzero_si256());
        let msk1 = _mm256_cmpeq_epi8(bit1, _mm256_setzero_si256());

        let mut res = _mm256_movemask_epi8(msk1) as u64;
        res = (res << 32) | _mm256_movemask_epi8(msk0) as u64;

        _tzcnt_u64(res) as usize
    }
}

#[cfg(target_feature = "sse4.2")]
#[inline]
pub fn _strspn_ascii_32_sse(buf: &[u8], arbm: __m128i) -> usize {
    debug_assert!(buf.len() >= 32);

    let ptr = buf.as_ptr();

    #[allow(non_snake_case, overflowing_literals)]
    unsafe {
        let LSH: __m128i = _mm_set1_epi8(0x0f);
        let ARF: __m128i = _mm_setr_epi8(
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        );

        let dat0 = _mm_lddqu_si128(ptr as *const _);
        let dat1 = _mm_lddqu_si128(ptr.offset(16) as *const _);

        let rbm0 = _mm_shuffle_epi8(arbm, dat0);
        let rbm1 = _mm_shuffle_epi8(arbm, dat1);

        let col0 = _mm_and_si128(LSH, _mm_srli_epi16(dat0, 4));
        let col1 = _mm_and_si128(LSH, _mm_srli_epi16(dat1, 4));

        let bit0 = _mm_and_si128(_mm_shuffle_epi8(ARF, col0), rbm0);
        let bit1 = _mm_and_si128(_mm_shuffle_epi8(ARF, col1), rbm1);

        let msk0 = _mm_cmpeq_epi8(bit0, _mm_setzero_si128());
        let msk1 = _mm_cmpeq_epi8(bit1, _mm_setzero_si128());

        let mut res = _mm_movemask_epi8(msk1) as u32;
        res = (res << 16) | _mm_movemask_epi8(msk0) as u32;

        _tzcnt_u32(res) as usize
    }
}

#[cfg(target_feature = "sse4.2")]
#[inline]
pub fn _strspn_ascii_64_sse(buf: &[u8], arbm: __m128i) -> usize {
    debug_assert!(buf.len() >= 64);

    let ptr = buf.as_ptr();

    #[allow(non_snake_case, overflowing_literals)]
    unsafe {
        let LSH: __m128i = _mm_set1_epi8(0x0f);
        let ARF: __m128i = _mm_setr_epi8(
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        );

        let dat0 = _mm_lddqu_si128(ptr as *const _);
        let dat1 = _mm_lddqu_si128(ptr.offset(16) as *const _);
        let dat2 = _mm_lddqu_si128(ptr.offset(32) as *const _);
        let dat3 = _mm_lddqu_si128(ptr.offset(48) as *const _);

        let rbm0 = _mm_shuffle_epi8(arbm, dat0);
        let rbm1 = _mm_shuffle_epi8(arbm, dat1);
        let rbm2 = _mm_shuffle_epi8(arbm, dat2);
        let rbm3 = _mm_shuffle_epi8(arbm, dat3);

        let col0 = _mm_and_si128(LSH, _mm_srli_epi16(dat0, 4));
        let col1 = _mm_and_si128(LSH, _mm_srli_epi16(dat1, 4));
        let col2 = _mm_and_si128(LSH, _mm_srli_epi16(dat2, 4));
        let col3 = _mm_and_si128(LSH, _mm_srli_epi16(dat3, 4));

        let bit0 = _mm_and_si128(_mm_shuffle_epi8(ARF, col0), rbm0);
        let bit1 = _mm_and_si128(_mm_shuffle_epi8(ARF, col1), rbm1);
        let bit2 = _mm_and_si128(_mm_shuffle_epi8(ARF, col2), rbm2);
        let bit3 = _mm_and_si128(_mm_shuffle_epi8(ARF, col3), rbm3);

        let msk0 = _mm_cmpeq_epi8(bit0, _mm_setzero_si128());
        let msk1 = _mm_cmpeq_epi8(bit1, _mm_setzero_si128());
        let msk2 = _mm_cmpeq_epi8(bit2, _mm_setzero_si128());
        let msk3 = _mm_cmpeq_epi8(bit3, _mm_setzero_si128());

        let mut res = _mm_movemask_epi8(msk3) as u64;
        res = (res << 16) | _mm_movemask_epi8(msk2) as u64;
        res = (res << 16) | _mm_movemask_epi8(msk1) as u64;
        res = (res << 16) | _mm_movemask_epi8(msk0) as u64;

        _tzcnt_u64(res) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_feature = "avx2")]
    #[test]
    fn test_strspn_ascii_32_avx() {
        // Only allow 'A' (0x41)
        let allow = unsafe { _mm256_setr_epi8(
            0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ) };

        let mut buf = vec![0x41; 31];
        buf.push(0x00);

        assert_eq!(_strspn_ascii_32_avx(&buf, allow), 31);
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn test_strspn_ascii_64_avx() {
        // Only allow 'B' (0x42)
        let allow = unsafe { _mm256_setr_epi8(
            0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ) };

        let mut buf = vec![0x42; 63];
        buf.push(0x00);

        assert_eq!(_strspn_ascii_64_avx(&buf, allow), 63);
    }

    #[cfg(target_feature = "sse4.2")]
    #[test]
    fn test_strspn_ascii_32_sse() {
        // Only allow 'A' (0x41)
        let allow = unsafe { _mm_setr_epi8(
            0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ) };

        let mut buf = vec![0x41; 31];
        buf.push(0x00);

        assert_eq!(_strspn_ascii_32_sse(&buf, allow), 31);
    }

    #[cfg(target_feature = "sse4.2")]
    #[test]
    fn test_strspn_ascii_64_sse() {
        // Only allow 'B' (0x42)
        let allow = unsafe { _mm_setr_epi8(
            0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ) };

        let mut buf = vec![0x42; 63];
        buf.push(0x00);

        assert_eq!(_strspn_ascii_64_sse(&buf, allow), 63);
    }
}
