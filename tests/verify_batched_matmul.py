"""验证 batched matmul 新路径的正确性（一次性脚本，跑完可删）。

覆盖：
- (3,3) 等批 / (2,3) / (3,1) / (1,3)
- 批维广播 (1,3,4,5)@(3,4,5)、(3,1,4)@(3,4,5)
- 复数 batched
- 错误路径：批维不匹配、内维不匹配
- fallback 边界：b (1,k) k>1 仍走 Python fallback（行为不变）
- choose 预计算重构的行为一致性
"""
import rsarray as np

fail = []


def check(name, got, want):
    g = np.asarray(got).tolist()
    w = np.asarray(want).tolist()
    if g != w:
        fail.append(name)
        print(f"FAIL {name}: got {g!r:.200} want {w!r:.200}")
    else:
        print(f"ok   {name}")


def near(name, got, want, tol=1e-9):
    g = np.asarray(got).ravel().tolist()
    w = np.asarray(want).ravel().tolist()
    if len(g) != len(w) or any(abs(a - b) > tol for a, b in zip(g, w)):
        fail.append(name)
        print(f"FAIL {name}: got {g!r:.300} want {w!r:.300}")
    else:
        print(f"ok   {name}")


# 基线：逐批 matmul（独立参考路径，处理 1-D 边与批侧）
def ref_batched(a, b):
    na, nb = a.ndim, b.ndim
    if na == 1 and nb == 1:
        return [a @ b]
    if na >= 2 and nb == 1:
        return [(a[i] @ b).tolist() for i in range(a.shape[0])]
    if na == 1 and nb >= 2:
        return [(a @ b[i]).tolist() for i in range(b.shape[0])]
    return [(a[i] @ b[i]).tolist() for i in range(a.shape[0])]


A = np.array([[[1.0, 2.0], [3.0, 4.0]],
              [[5.0, 6.0], [7.0, 8.0]]])          # (2,2,2)
B = np.array([[[9.0, 8.0], [7.0, 6.0]],
              [[5.0, 4.0], [3.0, 2.0]]])          # (2,2,2)
C = np.array([[[1.0, 2.0, 3.0]],
              [[4.0, 5.0, 6.0]]])                 # (2,1,3)
D = np.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])  # (2,3,1)
V = np.array([1.0, 2.0, 3.0])                     # (3,)

# (3,3) 等批
check("(3,3) 等批", A @ B, ref_batched(A, B))
# (2,3)
check("(2,3)", C @ D, ref_batched(C, D))
# (3,2) 等批（Rust 原有快路径，回归确认）
V2 = np.array([1.0, 2.0])                             # (2,)
check("(3,2) 等批", A @ V2, ref_batched(A, V2))
# (3,1) 矩阵×向量批：C 内维 3 → 向量须长 3
check("(3,1) 矩阵×向量批", C @ V, ref_batched(C, V))
# (1,3) 向量×矩阵批：D 内维 3 → (3,)@(2,3,1) → (2,1)
check("(1,3) 向量×矩阵批", V @ D, ref_batched(V, D))
# 批维广播：a 批=1，b 批=3
Ab1 = A[:1]                                        # (1,2,2)
B3 = np.array(B.tolist() * 2) if False else np.array(
    [B[0].tolist(), B[1].tolist(), A[0].tolist()])  # (3,2,2)
check("广播 (1,..)@(3,..)", Ab1 @ B3, ref_batched(np.array(Ab1.tolist() * 3), B3))
# (3,1,k) 形状 m=1
check("m=1 批 (3,1,4)@(3,4,5)",
      np.array([[[1.0, 2.0, 3.0, 4.0]]]) @ D if False else
      np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]) @
      np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]),
      ref_batched(np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]),
                  np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])))

# 复数 batched
Aci = np.array([[1.0 + 1.0j, 2.0], [0.0, 1.0 - 1.0j]])
Bci = np.array([[1.0, 0.5 + 0.5j], [1.0 + 1.0j, 1.0]])
Acb = np.array([Aci.tolist(), Aci.tolist()])
Bcb = np.array([Bci.tolist(), Bci.tolist()])
near("复数 batched (3,3)", Acb @ Bcb, ref_batched(Acb, Bcb), tol=1e-9)

# 错误路径
try:
    np.array([[[1.0, 2.0]], [[3.0, 4.0]]]) @ B3   # 批 2 vs 3 → 须报错
    fail.append("批不匹配未报错")
    print("FAIL 批不匹配未报错")
except ValueError:
    print("ok   批不匹配 → ValueError")
try:
    C @ np.array([[[1.0], [2.0]]])                 # 内维 3 vs 1 → 须报错
    fail.append("内维不匹配未报错")
    print("FAIL 内维不匹配未报错")
except ValueError:
    print("ok   内维不匹配 → ValueError")

# fallback 边界：(B,m,k) @ (1,k) k>1 保持原 fallback 语义
F = np.array([[[1.0, 2.0], [3.0, 4.0]]])
Row = np.array([[10.0, 20.0]])
got = (F @ Row).tolist()
# fallback 特判语义：(B,m,k)@(1,k) → (B,m,1)（rsarray 既有行为，保持）
want = [[[1.0 * 10 + 2.0 * 20], [3.0 * 10 + 4.0 * 20]]]
check("fallback 边界 (1,k) 外积语义", got, want)

# choose 重构一致性
idx = np.array([0, 2, 1, 0, 2, 1])
choices = [np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
           np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
           np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])]
manual = [choices[k][i] for i, k in enumerate(idx.tolist())]
check("choose 数组 choices", idx.choose(choices), manual)
check("choose 标量 choices", idx.choose([7, 8, 9]), [7, 9, 8, 7, 9, 8])

print()
print("ALL PASS" if not fail else f"FAILED: {fail}")
raise SystemExit(1 if fail else 0)
