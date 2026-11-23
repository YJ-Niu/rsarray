# SPDX-License-Identifier: MIT
"""位生成器 Rust 化回归验证（非门禁，手动运行）。

对照对象：纯 Python 参考实现（本文件内独立实现，与 Rust 化前的
_MT19937Legacy / PCG64 逻辑一致）——逐位（bit-exact）比较：
  1. MT19937 流：res53 / rk_interval / 缓冲 u8/u16/bool / 极坐标正态
  2. PCG64 流：generic res53 / next32/next64 / random_interval / 缓冲 / 正态
  3. legacy RandomState API：rand/randint/normal/choice/permutation/shuffle/bytes
     /get_state/set_state（含 array 种子）
  4. PCG64 legacy（RandomState(PCG64())）API
  5. 现代 Generator.shuffle（PCG64 流位级）、permutation（复现性）
  6. 性能对比（纯 Python 参考 vs Rust 批量）

运行：PYTHONPATH=python .venv/bin/python tests/verify_random_streams.py
"""
import math
import time

import rsarray as np
from rsarray.random import _SeedSequence

# ============================================================================
# 纯 Python 参考实现（独立于被测 Rust 路径）
# ============================================================================

MASK128 = (1 << 128) - 1
MASK64 = (1 << 64) - 1
MULT128 = (2549297995355413924 << 64) | 4865540595714422341


def gen_mask(m):
    mask = m
    for s in (1, 2, 4, 8, 16, 32):
        mask |= mask >> s
    return mask


class RefMT19937:
    """标准 MT19937（init_genrand / init_by_array / twist / tempering）。"""

    def __init__(self, seed):
        if isinstance(seed, (list, tuple, range)):
            self._init_by_array(seed)
            return
        seed = int(seed) & 0xffffffff
        mt = [0] * 624
        mt[0] = seed
        for i in range(1, 624):
            mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & 0xffffffff
        self._mt = mt
        self._mti = 624
        self._gauss_cache = None

    def _init_by_array(self, key):
        key = [int(k) & 0xffffffff for k in key]
        key_len = len(key)
        mt = [0] * 624
        mt[0] = 19650218
        for i in range(1, 624):
            mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & 0xffffffff
        i, j = 1, 0
        k = 624 if 624 > key_len else key_len
        for _ in range(k):
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525)) + key[j] + j
            mt[i] &= 0xffffffff
            i += 1
            j += 1
            if i >= 624:
                mt[0] = mt[623]
                i = 1
            if j >= key_len:
                j = 0
        for _ in range(623, 0, -1):
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941)) - i
            mt[i] &= 0xffffffff
            i += 1
            if i >= 624:
                mt[0] = mt[623]
                i = 1
        mt[0] = 0x80000000
        self._mt = mt
        self._mti = 624
        self._gauss_cache = None

    def _twist(self):
        mt = self._mt
        for i in range(624):
            y = (mt[i] & 0x80000000) + (mt[(i + 1) % 624] & 0x7fffffff)
            v = mt[(i + 397) % 624] ^ (y >> 1)
            if y & 1:
                v ^= 0x9908b0df
            mt[i] = v & 0xffffffff
        self._mti = 0

    def u32(self):
        if self._mti >= 624:
            self._twist()
        x = self._mt[self._mti]
        self._mti += 1
        x ^= (x >> 11)
        x ^= (x << 7) & 0x9d2c5680
        x ^= (x << 15) & 0xefc60000
        x ^= (x >> 18)
        return x & 0xffffffff

    def random_sample(self):
        a = self.u32() >> 5
        b = self.u32() >> 6
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)

    def random_samples(self, n):
        return [self.random_sample() for _ in range(n)]

    def interval(self, maxv):
        if maxv == 0:
            return 0
        mask = (1 << maxv.bit_length()) - 1
        while True:
            v = self.u32() & mask
            if v <= maxv:
                return v

    def intervals(self, rngs):
        return [self.interval(r) for r in rngs]

    def intervals_u8(self, rngs):
        buf = 0
        bcnt = 0
        out = []
        for rng in rngs:
            if rng == 0:
                out.append(0)
                continue
            mask = gen_mask(rng)
            while True:
                if bcnt == 0:
                    buf = self.u32()
                    bcnt = 3
                else:
                    buf >>= 8
                    bcnt -= 1
                val = buf & 0xFF & mask
                if val <= rng:
                    break
            out.append(val)
        return out

    def intervals_u16(self, rngs):
        buf = 0
        bcnt = 0
        out = []
        for rng in rngs:
            if rng == 0:
                out.append(0)
                continue
            mask = gen_mask(rng)
            while True:
                if bcnt == 0:
                    buf = self.u32()
                    bcnt = 1
                else:
                    buf >>= 16
                    bcnt -= 1
                val = buf & 0xFFFF & mask
                if val <= rng:
                    break
            out.append(val)
        return out

    def intervals_bool(self, rngs):
        buf = 0
        bcnt = 0
        out = []
        for rng in rngs:
            if rng == 0:
                out.append(0)
                continue
            if bcnt == 0:
                buf = self.u32()
                bcnt = 31
            else:
                buf >>= 1
                bcnt -= 1
            out.append(1 if (buf & 1) else 0)
        return out

    def standard_normal(self):
        if self._gauss_cache is not None:
            x = self._gauss_cache
            self._gauss_cache = None
            return x
        while True:
            v1 = 2.0 * self.random_sample() - 1.0
            v2 = 2.0 * self.random_sample() - 1.0
            s = v1 * v1 + v2 * v2
            if 0.0 < s < 1.0:
                break
        r = math.sqrt(-2.0 * math.log(s) / s)
        self._gauss_cache = v1 * r
        return v2 * r

    def standard_normals(self, n):
        return [self.standard_normal() for _ in range(n)]


class RefPCG64:
    """PCG XSL-RR-128-64 核心流（播种链复用项目 _SeedSequence，
    该部分未改动且已逐位对照标准）。"""

    def __init__(self, seed):
        ss = _SeedSequence(seed)
        self._seed_seq = ss
        val = ss.generate_state(4, 'uint64')
        initstate = (val[0] << 64) | val[1]
        initseq = (val[2] << 64) | val[3]
        self._state = 0
        self._inc = ((initseq << 1) | 1) & MASK128
        self._step()
        self._state = (self._state + initstate) & MASK128
        self._step()
        self._has_uint32 = 0
        self._uinteger = 0

    def _step(self):
        self._state = (self._state * MULT128 + self._inc) & MASK128

    def next64(self):
        self._step()
        hi = self._state >> 64
        lo = self._state & MASK64
        x = hi ^ lo
        rot = (self._state >> 122) & 63
        return ((x >> rot) | (x << ((-rot) & 63))) & MASK64

    def next32(self):
        if self._has_uint32:
            self._has_uint32 = 0
            return self._uinteger
        n = self.next64()
        self._has_uint32 = 1
        self._uinteger = n >> 32
        return n & 0xFFFFFFFF

    def random_sample(self):
        return (self.next64() >> 11) * (1.0 / 9007199254740992.0)

    def random_samples(self, n):
        return [self.random_sample() for _ in range(n)]

    def interval(self, maxv):
        """random_interval（distributions.c）：掩码拒绝；
        maxv ≤ 2^32-1 走 next32 流，否则 next64 流。"""
        if maxv == 0:
            return 0
        mask = gen_mask(maxv)
        if maxv <= 0xffffffff:
            while True:
                v = self.next32() & mask
                if v <= maxv:
                    return v
        else:
            while True:
                v = self.next64() & mask
                if v <= maxv:
                    return v

    def intervals(self, rngs):
        return [self.interval(r) for r in rngs]

    def intervals_u8(self, rngs):
        buf = 0
        bcnt = 0
        out = []
        for rng in rngs:
            if rng == 0:
                out.append(0)
                continue
            mask = gen_mask(rng)
            while True:
                if bcnt == 0:
                    buf = self.next32()
                    bcnt = 3
                else:
                    buf >>= 8
                    bcnt -= 1
                val = buf & 0xFF & mask
                if val <= rng:
                    break
            out.append(val)
        return out

    def intervals_u16(self, rngs):
        buf = 0
        bcnt = 0
        out = []
        for rng in rngs:
            if rng == 0:
                out.append(0)
                continue
            mask = gen_mask(rng)
            while True:
                if bcnt == 0:
                    buf = self.next32()
                    bcnt = 1
                else:
                    buf >>= 16
                    bcnt -= 1
                val = buf & 0xFFFF & mask
                if val <= rng:
                    break
            out.append(val)
        return out

    def intervals_bool(self, rngs):
        buf = 0
        bcnt = 0
        out = []
        for rng in rngs:
            if rng == 0:
                out.append(0)
                continue
            if bcnt == 0:
                buf = self.next32()
                bcnt = 31
            else:
                buf >>= 1
                bcnt -= 1
            out.append(1 if (buf & 1) else 0)
        return out

    def standard_normal(self):
        if getattr(self, '_gauss_cache', None) is not None:
            x = self._gauss_cache
            self._gauss_cache = None
            return x
        while True:
            v1 = 2.0 * self.random_sample() - 1.0
            v2 = 2.0 * self.random_sample() - 1.0
            s = v1 * v1 + v2 * v2
            if 0.0 < s < 1.0:
                break
        r = math.sqrt(-2.0 * math.log(s) / s)
        self._gauss_cache = v1 * r
        return v2 * r

    def standard_normals(self, n):
        out = []
        for _ in range(n):
            out.append(self.standard_normal())
        return out

    def state_dict(self):
        return {'bit_generator': 'PCG64',
                'state': {'state': self._state, 'inc': self._inc},
                'has_uint32': self._has_uint32,
                'uinteger': self._uinteger}

    @classmethod
    def from_state(cls, state, inc):
        """从已播种状态构造（seed=None 时熵非确定，须共享同一初始状态）。"""
        obj = cls.__new__(cls)
        obj._seed_seq = None
        obj._state = state
        obj._inc = inc
        obj._has_uint32 = 0
        obj._uinteger = 0
        return obj


# 参考 Fisher-Yates（legacy 语义：i 从 n-1 递减到 1，j = interval(i)）
def ref_fisher_yates(stream, n):
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = stream.interval(i)
        idx[i], idx[j] = idx[j], idx[i]
    return idx


# ============================================================================
# 测试框架
# ============================================================================

PASS = 0
FAIL = 0


def check(name, ok, detail=''):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {name} {detail}")


def tolist(x):
    if hasattr(x, 'tolist'):
        return x.tolist()
    return list(x)


def close(a, b, tol=1e-8):
    """逐元素近似比较（硬编码标准值只保留 8 位小数）。"""
    return len(a) == len(b) and all(
        math.isclose(x, y, rel_tol=tol, abs_tol=tol) for x, y in zip(a, b))


def _ss_right(cdf, u):
    """首个 cdf[i] > u 的 i（searchsorted side='right' 语义）。"""
    for i in range(len(cdf)):
        if cdf[i] > u:
            return i
    return len(cdf) - 1


def _norm_cdf(probs):
    total = 0.0
    for v in probs:
        total += v
    cum = np.cumsum(np.asarray(probs, dtype='float64')).ravel().tolist()
    return [c / total for c in cum]


# ============================================================================
# A. MT19937 流位级对照
# ============================================================================

MT_SEEDS = [0, 42, (1 << 32) - 1, [0], list(range(10)), [4294967295]]

MIXED_MAXV = [0, 1, 2, 7, 8, 100, 255, 256, 1000, 65535, 65536, (1 << 32) - 1]
U8_RNGS = [0, 1, 2, 127, 128, 200, 255]
U16_RNGS = [0, 1, 2, 1000, 65534, 65535]
BOOL_RNGS = [0, 1]


def test_mt_streams():
    """播种留在 Python（已对照标准），交接后 Rust 流与参考流逐位一致。
    ref 与 rs 的调用顺序严格锁步，保证流状态对齐。"""
    import rsarray.num_core as c
    for seed in MT_SEEDS:
        ref = RefMT19937(seed)
        rs = c.random.Mt19937Stream(list(ref._mt), ref._mti)
        tag = f"MT seed={seed if isinstance(seed, int) else 'arr'}"

        check(f"{tag} random_samples",
              rs.random_samples(4000) == ref.random_samples(4000))

        check(f"{tag} intervals",
              rs.intervals(MIXED_MAXV * 400) == ref.intervals(MIXED_MAXV * 400))

        check(f"{tag} intervals_u8",
              rs.intervals_u8(U8_RNGS * 3000) == ref.intervals_u8(U8_RNGS * 3000))

        check(f"{tag} intervals_u16",
              rs.intervals_u16(U16_RNGS * 2000) == ref.intervals_u16(U16_RNGS * 2000))

        check(f"{tag} intervals_bool",
              rs.intervals_bool(BOOL_RNGS * 2000)
              == ref.intervals_bool(BOOL_RNGS * 2000))

        check(f"{tag} standard_normals",
              rs.standard_normals(3000) == ref.standard_normals(3000))

        # 缓存跨调用：2500 + 500 == 3000（ref 与 rs 锁步延续）
        a = rs.standard_normals(2500)
        b = rs.standard_normals(500)
        check(f"{tag} normal cache carryover",
              a + b == ref.standard_normals(3000))


# ============================================================================
# B. PCG64 流位级对照
# ============================================================================

PCG_SEEDS = [None, 0, 42, [1, 2, 3], 12345]
PCG_MAXV = [0, 1, 2, 7, 100, 255, 256, 1000, 65535, 65536, (1 << 32) - 1,
            1 << 40, (1 << 63) - 1]


def test_pcg_streams():
    """每项子检查使用一对全新流（从同一初始状态构造）——seed=None 时
    熵非确定，必须共享首次播种的初始状态。"""
    import rsarray.num_core as c
    for seed in PCG_SEEDS:
        ref0 = RefPCG64(seed)
        s0, i0 = ref0._state, ref0._inc
        tag = f"PCG seed={seed if seed is not None else 'None'}"

        def ref_fresh():
            return RefPCG64.from_state(s0, i0)

        def rs_fresh():
            return c.random.Pcg64Stream(s0, i0, 0, 0)

        ref, rs = ref_fresh(), rs_fresh()
        check(f"{tag} random_samples",
              rs.random_samples(4000) == ref.random_samples(4000))

        ref, rs = ref_fresh(), rs_fresh()
        check(f"{tag} next32",
              [rs.next32_one() for _ in range(2000)]
              == [ref.next32() for _ in range(2000)])

        ref, rs = ref_fresh(), rs_fresh()
        check(f"{tag} next64",
              [rs.next64_one() for _ in range(2000)]
              == [ref.next64() for _ in range(2000)])

        ref, rs = ref_fresh(), rs_fresh()
        check(f"{tag} intervals",
              rs.intervals(PCG_MAXV * 200) == ref.intervals(PCG_MAXV * 200))

        ref, rs = ref_fresh(), rs_fresh()
        check(f"{tag} intervals_u8",
              rs.intervals_u8(U8_RNGS * 3000)
              == ref.intervals_u8(U8_RNGS * 3000))

        ref, rs = ref_fresh(), rs_fresh()
        check(f"{tag} intervals_u16",
              rs.intervals_u16(U16_RNGS * 2000)
              == ref.intervals_u16(U16_RNGS * 2000))

        ref, rs = ref_fresh(), rs_fresh()
        check(f"{tag} intervals_bool",
              rs.intervals_bool(BOOL_RNGS * 2000)
              == ref.intervals_bool(BOOL_RNGS * 2000))

        ref, rs = ref_fresh(), rs_fresh()
        check(f"{tag} standard_normals",
              rs.standard_normals(3000) == ref.standard_normals(3000))

        # 缓存跨调用：2500 + 500 == 3000
        ref, rs = ref_fresh(), rs_fresh()
        a = rs.standard_normals(2500)
        b = rs.standard_normals(500)
        check(f"{tag} normal cache carryover",
              a + b == ref.standard_normals(3000))

        # state 往返：抽取 100 个后状态 == 参考状态；恢复后流延续
        ref, rs = ref_fresh(), rs_fresh()
        ref.random_samples(100)
        rs.random_samples(100)
        s_state, s_inc, s_h32, s_ue = rs.state()
        s_ref = ref.state_dict()
        check(f"{tag} state match",
              s_state == s_ref['state']['state']
              and s_inc == s_ref['state']['inc']
              and s_h32 == s_ref['has_uint32']
              and s_ue == s_ref['uinteger'])
        cont_ref = ref.random_samples(50)
        rs.set_state(s_state, s_inc, s_h32, s_ue)
        check(f"{tag} state restore", rs.random_samples(50) == cont_ref)


# ============================================================================
# C. legacy RandomState API 位级对照
# ============================================================================

# 已知标准值（seed 42，跨版本稳定）
NP42_RAND = [0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864,
             0.15599452, 0.05808361, 0.86617615, 0.60111501, 0.70807258]
NP42_RANDINT = [6, 3, 7, 4, 6, 9, 2, 6, 7, 4]
NP42_NORMAL = [0.49671415, -0.1382643, 0.64768854, 1.52302986, -0.23415337,
               -0.23413696, 1.57921282, 0.76743473, -0.46947439, 0.54256004]


def test_legacy_api():
    # 已知标准值（锚定 numpy；浮点常量 8 位小数，用近似比较）
    rs = np.random.RandomState(42)
    check("legacy rand(10) == numpy-42", close(tolist(rs.rand(10)), NP42_RAND))
    rs = np.random.RandomState(42)
    _ri = tolist(rs.randint(0, 10, 10))
    check("legacy randint(0,10,10) == numpy-42", _ri == NP42_RANDINT,
          f"got={_ri}")
    rs = np.random.RandomState(42)
    check("legacy normal(0,1,10) == numpy-42",
          close(tolist(rs.normal(0, 1, 10)), NP42_NORMAL))

    # 与参考流对照（每次 API 调用 = 一个全新流）
    ref = RefMT19937(123)
    rs = np.random.RandomState(123)
    check("legacy rand(1000)", tolist(rs.rand(1000)) == ref.random_samples(1000))

    ref = RefMT19937(123)
    rs = np.random.RandomState(123)
    check("legacy randint(0,1000,1000)",
          tolist(rs.randint(0, 1000, 1000))
          == [v for v in ref.intervals([999] * 1000)])

    ref = RefMT19937(123)
    rs = np.random.RandomState(123)
    check("legacy randint uint8",
          tolist(rs.randint(0, 256, 500, dtype='uint8'))
          == ref.intervals_u8([255] * 500))

    ref = RefMT19937(123)
    rs = np.random.RandomState(123)
    check("legacy randint int16 signed",
          tolist(rs.randint(-5, 5, 500, dtype='int16'))
          == [-5 + v for v in ref.intervals_u16([9] * 500)])

    ref = RefMT19937(123)
    rs = np.random.RandomState(123)
    check("legacy randint uint16",
          tolist(rs.randint(0, 1000, 500, dtype='uint16'))
          == ref.intervals_u16([999] * 500))

    ref = RefMT19937(123)
    rs = np.random.RandomState(123)
    check("legacy randint 64-bit general",
          tolist(rs.randint(0, 1 << 62, 200, dtype='uint64'))
          == ref.intervals([(1 << 62) - 1] * 200))

    ref = RefMT19937(123)
    rs = np.random.RandomState(123)
    check("legacy normal(3,2,1000)",
          tolist(rs.normal(3, 2, 1000))
          == [3 + 2 * v for v in ref.standard_normals(1000)])

    ref = RefMT19937(123)
    rs = np.random.RandomState(123)
    check("legacy choice replace",
          tolist(rs.choice(50, 300)) == ref.intervals([49] * 300))

    ref = RefMT19937(123)
    rs = np.random.RandomState(123)
    check("legacy choice !replace",
          tolist(rs.choice(50, 20, replace=False))
          == ref_fisher_yates(ref, 50)[:20])

    # choice with p（cdf 语义与实现一致：np.cumsum 后归一化）
    pl = [0.1, 0.2, 0.3, 0.4]
    cdf = _norm_cdf(pl)
    ref = RefMT19937(123)
    rs = np.random.RandomState(123)
    vals = [10, 20, 30, 40]
    out_ref = [_ss_right(cdf, u) for u in ref.random_samples(100)]
    check("legacy choice with p",
          tolist(rs.choice(np.asarray(vals), 100, p=np.asarray(pl)))
          == [vals[i] for i in out_ref])

    ref = RefMT19937(123)
    rs = np.random.RandomState(123)
    check("legacy permutation",
          tolist(rs.permutation(100)) == ref_fisher_yates(ref, 100))

    ref = RefMT19937(123)
    rs = np.random.RandomState(123)
    x = list(range(100))
    rs.shuffle(x)
    check("legacy shuffle list", x == ref_fisher_yates(ref, 100))

    ref = RefMT19937(123)
    rs = np.random.RandomState(123)
    b_ref = b''.join(v.to_bytes(4, 'little')
                     for v in ref.intervals([0xffffffff] * 16))[:64]
    check("legacy bytes(64)", rs.bytes(64) == b_ref)

    ref = RefMT19937(123)
    rs = np.random.RandomState(123)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        _ri = tolist(rs.random_integers(5, 500, 500))
    check("legacy random_integers",
          _ri == [5 + v for v in ref.intervals([495] * 500)])

    # get_state / set_state 往返
    rs = np.random.RandomState(777)
    a = tolist(rs.rand(10))
    st = rs.get_state()
    check("get_state format", st[0] == 'MT19937' and len(st[1]) == 624
          and len(st) == 5)
    b1 = tolist(rs.rand(10))
    rs.set_state(st)
    b2 = tolist(rs.rand(10))
    check("set_state roundtrip", b1 == b2)
    check("get_state values vs ref", _mt_state_equals(st, RefMT19937(777), 10, a))

    # array 种子
    ref = RefMT19937(range(10))
    rs = np.random.RandomState(range(10))
    check("array seed rand(100)",
          tolist(rs.rand(100)) == ref.random_samples(100))

    # 校验错误不消耗流
    rs = np.random.RandomState(5)
    ref = RefMT19937(5)
    v1 = rs.rand(1)
    r1 = ref.random_samples(1)
    for fn in (lambda: rs.uniform(1.0, 0.0), lambda: rs.uniform(0.0, 0.0),
               lambda: rs.normal(0.0, -1.0),
               lambda: rs.normal(0.0, -0.0),
               lambda: rs.uniform(0.0, float('inf'))):
        try:
            fn()
            check("uniform/normal validation raises", False)
            return
        except (ValueError, OverflowError):
            pass
    v2 = tolist(rs.rand(1))
    r2 = ref.random_samples(1)
    check("validation errors consume no stream", v1 == r1 and v2 == r2)

    # 模块级 uniform/normal fast-path（全局 legacy 状态）
    np.random.seed(99)
    ref = RefMT19937(99)
    check("module uniform fast-path",
          tolist(np.random.uniform(2.0, 5.0, 500))
          == [2.0 + (5.0 - 2.0) * u for u in ref.random_samples(500)])
    np.random.seed(99)
    ref = RefMT19937(99)
    check("module normal fast-path",
          tolist(np.random.normal(1.0, 0.5, 500))
          == [1.0 + 0.5 * v for v in ref.standard_normals(500)])
    np.random.seed(99)
    ref = RefMT19937(99)
    check("module uniform scalar",
          np.random.uniform(2.0, 5.0) == 2.0 + (5.0 - 2.0) * ref.random_sample())


def _mt_state_equals(st, ref, draws, _a):
    """get_state 的 key/pos 与参考（同样抽取 draws 个后）一致。"""
    ref.random_samples(draws)
    return list(st[1]) == ref._mt and st[2] == ref._mti


# ============================================================================
# D. PCG64 legacy（RandomState(PCG64())）API 位级对照
# ============================================================================

def test_pcg64_legacy_api():
    bg = np.random.PCG64(7)
    rs = np.random.RandomState(bg)
    ref = RefPCG64(7)
    check("pcg-legacy rand(1000)",
          tolist(rs.rand(1000)) == ref.random_samples(1000))

    ref = RefPCG64(7)
    rs = np.random.RandomState(np.random.PCG64(7))
    check("pcg-legacy randint uint8",
          tolist(rs.randint(0, 256, 500, dtype='uint8'))
          == ref.intervals_u8([255] * 500))

    ref = RefPCG64(7)
    rs = np.random.RandomState(np.random.PCG64(7))
    check("pcg-legacy randint int16 signed",
          tolist(rs.randint(-5, 5, 500, dtype='int16'))
          == [-5 + v for v in ref.intervals_u16([9] * 500)])

    ref = RefPCG64(7)
    rs = np.random.RandomState(np.random.PCG64(7))
    check("pcg-legacy randint uint16",
          tolist(rs.randint(0, 1000, 500, dtype='uint16'))
          == ref.intervals_u16([999] * 500))

    # 默认 int64：通用 interval 路径（maxv ≤ 2^32 走 next32 流）
    ref = RefPCG64(7)
    rs = np.random.RandomState(np.random.PCG64(7))
    check("pcg-legacy randint int64 general",
          tolist(rs.randint(0, 1000, 500))
          == ref.intervals([999] * 500))

    # 64 位值域：next64 流分支（distributions.c 语义）。
    # 注：rsarray 数组以 f64 存储，>2^53 的整数值经 asarray 后按 f64 舍入
    # （核心存储限制，非流 bug）——流正确性已由 PCG intervals 流级测试覆盖，
    # 这里对照 f64 舍入后的值。
    ref = RefPCG64(7)
    rs = np.random.RandomState(np.random.PCG64(7))
    got = tolist(rs.randint(0, 1 << 62, 200, dtype='uint64'))
    refv = ref.intervals([(1 << 62) - 1] * 200)
    check("pcg-legacy randint 64-bit next64 branch",
          [int(float(v)) for v in got] == [int(float(v)) for v in refv])
    check("pcg-legacy 64-bit in range",
          all(0 <= v < (1 << 62) for v in refv))

    ref = RefPCG64(7)
    rs = np.random.RandomState(np.random.PCG64(7))
    check("pcg-legacy normal(3,2,500)",
          tolist(rs.normal(3, 2, 500))
          == [3 + 2 * v for v in ref.standard_normals(500)])

    ref = RefPCG64(7)
    rs = np.random.RandomState(np.random.PCG64(7))
    check("pcg-legacy permutation",
          tolist(rs.permutation(100)) == ref_fisher_yates(ref, 100))

    # state 一致性：bg.state（Rust）== 参考 state dict
    bg2 = np.random.PCG64(7)
    rs2 = np.random.RandomState(bg2)
    ref2 = RefPCG64(7)
    rs2.rand(100)
    ref2.random_samples(100)
    st = bg2.state
    check("pcg-legacy bg.state == ref",
          st == ref2.state_dict())
    check("pcg-legacy get_state == bg.state",
          rs2.get_state() == ref2.state_dict())

    # PCG64 state 往返（经 wrapper set_state）
    st_saved = rs2.get_state()
    cont1 = tolist(rs2.rand(20))
    rs2.set_state(st_saved)
    cont2 = tolist(rs2.rand(20))
    check("pcg-legacy set_state roundtrip", cont1 == cont2)


# ============================================================================
# E. 现代 Generator（shuffle 走 PCG64 位级流；permutation 走 Rust StdRng）
# ============================================================================

def test_modern_generator():
    # default_rng(seed).shuffle：PCG64(seed) 流位级对照
    g = np.random.default_rng(7)
    x = list(range(200))
    g.shuffle(x)
    check("modern shuffle list == PCG64(7) ref",
          x == ref_fisher_yates(RefPCG64(7), 200))

    # Generator(PCG64(seed)) 显式位生成器
    g2 = np.random.Generator(np.random.PCG64(9))
    y = list(range(200))
    g2.shuffle(y)
    check("modern shuffle explicit PCG64(9) == ref",
          y == ref_fisher_yates(RefPCG64(9), 200))

    # ndarray shuffle（axis 0）
    g3 = np.random.default_rng(5)
    a = np.arange(50, dtype='int64')
    g3.shuffle(a)
    # 参考：PCG64(5) Fisher-Yates 索引
    idx = ref_fisher_yates(RefPCG64(5), 50)
    expected = [i for i in idx]
    check("modern shuffle ndarray", tolist(a) == expected)

    # permutation：Rust StdRng 流（非位级），只验复现性 + 合法性
    p1 = tolist(np.random.default_rng(11).permutation(100))
    p2 = tolist(np.random.default_rng(11).permutation(100))
    check("modern permutation reproducible", p1 == p2)
    check("modern permutation valid", sorted(p1) == list(range(100)))

    # 核心分布（Rust 核，本轮未动）：复现性 + 值域
    g4 = np.random.default_rng(3)
    g5 = np.random.default_rng(3)
    check("modern random reproducible",
          tolist(g4.random(100)) == tolist(g5.random(100)))
    u1 = tolist(g4.uniform(0.0, 2.0, 100))
    u2 = tolist(g5.uniform(0.0, 2.0, 100))
    check("modern uniform reproducible + range",
          u1 == u2 and all(0.0 <= v < 2.0 for v in u1))
    i1 = tolist(g4.integers(0, 10, 100))
    i2 = tolist(g5.integers(0, 10, 100))
    check("modern integers reproducible + range",
          i1 == i2 and all(0 <= v < 10 for v in i1))
    n1 = tolist(g4.standard_normal(100))
    n2 = tolist(g5.standard_normal(100))
    check("modern standard_normal reproducible", n1 == n2)


# ============================================================================
# F. 性能基准：纯 Python 参考流 vs Rust 批量流
# ============================================================================

def bench(label, ref_fn, new_fn, n_check=None):
    t0 = time.perf_counter()
    rv = ref_fn()
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    nv = new_fn()
    t_new = time.perf_counter() - t0
    same = ''
    if n_check is not None:
        same = '  [values match]' if tolist(nv)[:n_check] == tolist(rv)[:n_check] \
            else '  [!! VALUES DIFFER !!]'
    speedup = t_ref / t_new if t_new > 0 else float('inf')
    print(f"  {label:<38} ref {t_ref:8.3f}s  new {t_new:8.4f}s  "
          f"x{speedup:8.1f}{same}")
    return t_ref, t_new


def test_benchmarks():
    print("\n性能基准（纯 Python 参考流 vs Rust 批量流）：")
    N = 1_000_000

    # 参考侧同样经 asarray 建数组（新路径返回 ndarray，须同口径比较）；
    # shuffle list 对比为原地操作，两侧均无需建数组。
    bench("MT rand(1M)",
          lambda: np.asarray(RefMT19937(42).random_samples(N), dtype='float64'),
          lambda: np.random.RandomState(42).rand(N), n_check=1000)

    bench("MT randint u8 (1M)",
          lambda: np.asarray(RefMT19937(42).intervals_u8([255] * N), dtype='int64'),
          lambda: np.random.RandomState(42).randint(0, 256, N, dtype='uint8'),
          n_check=1000)

    bench("MT randint int64 64-bit (500K)",
          lambda: np.asarray(RefMT19937(42).intervals([(1 << 62) - 1] * 500_000),
                           dtype='int64'),
          lambda: np.random.RandomState(42).randint(0, 1 << 62, 500_000),
          n_check=1000)

    bench("MT normal(0,1,1M)",
          lambda: np.asarray(RefMT19937(42).standard_normals(N), dtype='float64'),
          lambda: np.random.RandomState(42).normal(0, 1, N), n_check=1000)

    bench("MT permutation(500K)",
          lambda: np.asarray(ref_fisher_yates(RefMT19937(42), 500_000), dtype='int64'),
          lambda: np.random.RandomState(42).permutation(500_000),
          n_check=1000)

    bench("module uniform fast-path (1M)",
          lambda: np.asarray([0.0 + 1.0 * u
                              for u in RefMT19937(42).random_samples(N)],
                             dtype='float64'),
          lambda: _module_uniform(42, N), n_check=1000)

    bench("modern shuffle list (500K)",
          lambda: ref_fisher_yates(RefPCG64(7), 500_000),
          lambda: _modern_shuffle(7, 500_000), n_check=1000)

    bench("pcg-legacy rand(1M)",
          lambda: np.asarray(RefPCG64(7).random_samples(N), dtype='float64'),
          lambda: _pcg64_legacy_rand(7, N), n_check=1000)


def _module_uniform(seed, n):
    np.random.seed(seed)
    return np.random.uniform(0.0, 1.0, n)


def _modern_shuffle(seed, n):
    g = np.random.default_rng(seed)
    x = list(range(n))
    g.shuffle(x)
    return x


def _pcg64_legacy_rand(seed, n):
    return np.random.RandomState(np.random.PCG64(seed)).rand(n)


# ============================================================================
# 入口
# ============================================================================

def main():
    print("== MT19937 流位级对照 ==")
    test_mt_streams()
    print("== PCG64 流位级对照 ==")
    test_pcg_streams()
    print("== legacy RandomState API ==")
    test_legacy_api()
    print("== PCG64 legacy API ==")
    test_pcg64_legacy_api()
    print("== 现代 Generator ==")
    test_modern_generator()

    print(f"\n结果: {PASS} passed, {FAIL} failed")
    if FAIL:
        print("存在失败项！")
        return 1
    test_benchmarks()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
