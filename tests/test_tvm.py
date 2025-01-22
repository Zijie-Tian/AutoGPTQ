import tvm
import tvm.testing
from tvm import te # tensor expression
import numpy as np

from tvm.script import tir as T
from tvm import meta_schedule as ms
from tvm.script.parser.tir import evaluate

M = 1024
N = 1024
K = 1024

# define computation using tvm script
@tvm.script.ir_module
class MyMatMulModule:
  @T.prim_func
  def main(A: T.Buffer[(M, K), "float32"],
           B: T.Buffer[(K, N), "float32"],
           C: T.Buffer[(M, N), "float32"],
           ):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    for i, j, k in T.grid(M, N, K):
      with T.block("C"):
        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
        with T.init():
          C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

sch = tvm.tir.Schedule(MyMatMulModule)
block_C = sch.get_block("C")
i, j, k = sch.get_loops(block=block_C)
i0, i1 = sch.split(i, [None, 128])
sch.mod.show()

sch.bind(i0, "blockIdx.x")
sch.bind(i1, "threadIdx.x")
sch.mod.show()

num_flop = 2 * M * N * K
rt_mod = tvm.build(sch.mod, target="nvidia/nvidia-a30")
dev = tvm.cuda(0)
A_np = np.random.uniform(size=(M, K)).astype("float32")
B_np = np.random.uniform(size=(K, N)).astype("float32")
A_nd = tvm.nd.array(A_np, dev)
B_nd = tvm.nd.array(B_np, dev)
C_nd = tvm.nd.array(np.zeros((M, N), dtype="float32"), dev)
evaluator = rt_mod.time_evaluator("main", dev, number=10)
print("MetaSchedule: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))

# assert correctness
np.testing.assert_allclose(C_nd.numpy(), A_np @ B_np, rtol=1e-5, atol=1e-5)


def blocking(sch,
             tile_local_y,
             tile_local_x,
             tile_block_y,
             tile_block_x,
             tile_k):
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")

    i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])
    sch.unroll(k1)
    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")

    sch.bind(i1, "threadIdx.y")
    sch.bind(j1, "threadIdx.x")
    sch.decompose_reduction(block_C, k0)

    return sch

sch = tvm.tir.Schedule(MyMatMulModule)
sch = blocking(sch, 8, 8, 8, 8, 4)
sch.mod.show()

rt_mod = tvm.build(sch.mod, target="cuda")
dev = tvm.cuda(0)
A_np = np.random.uniform(size=(1024, 1024)).astype("float32")
B_np = np.random.uniform(size=(1024, 1024)).astype("float32")
A_nd = tvm.nd.array(A_np, dev)
B_nd = tvm.nd.array(B_np, dev)
C_nd = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)

num_flop = 2 * 1024 * 1024 * 1024
evaluator = rt_mod.time_evaluator("main", dev, number=10)

print("GEMM-Blocking: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))


def cache_read_and_coop_fetch(sch, block, nthread, read_idx, read_loc):
    read_cache = sch.cache_read(block=block, read_buffer_index=read_idx, storage_scope="shared")
    sch.compute_at(block=read_cache, loop=read_loc)
    # vectorized cooperative fetch
    inner0, inner1 = sch.get_loops(block=read_cache)[-2:]
    inner = sch.fuse(inner0, inner1)
    _, tx, vec = sch.split(loop=inner, factors=[None, nthread, 4])
    sch.vectorize(vec)
    sch.bind(tx, "threadIdx.x")


def blocking_with_shared(
    sch, 
    tile_local_y, 
    tile_local_x, 
    tile_block_y, 
    tile_block_x,
    tile_k):
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")

    i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])

    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")

    tx = sch.fuse(i1, j1)
    sch.bind(tx, "threadIdx.x")
    nthread = tile_block_y * tile_block_x
    cache_read_and_coop_fetch(sch, block_C, nthread, 0, k0)
    cache_read_and_coop_fetch(sch, block_C, nthread, 1, k0)    
    sch.decompose_reduction(block_C, k0)

    return sch

sch = tvm.tir.Schedule(MyMatMulModule)
sch = blocking_with_shared(sch, 8, 8, 8, 8, 8)
sch.mod.show()

rt_mod = tvm.build(sch.mod, target="cuda")
dev = tvm.cuda(0)
A_np = np.random.uniform(size=(M, K)).astype("float32")
B_np = np.random.uniform(size=(K, N)).astype("float32")
A_nd = tvm.nd.array(A_np, dev)
B_nd = tvm.nd.array(B_np, dev)
C_nd = tvm.nd.array(np.zeros((M, N), dtype="float32"), dev)

num_flop = 2 * M * N * K
evaluator = rt_mod.time_evaluator("main", dev, number=10)


print("GEMM-Blocking: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))


with open('matmul_tvm.h', 'w') as f:
    f.write(rt_mod.imported_modules[0].get_source())

database = ms.tune_tir(
    mod=MyMatMulModule,
    target="nvidia/nvidia-a30", # define target type
    work_dir="./tune_tmp",
    max_trials_global=64,
    num_trials_per_iter=64,
)

sch_tuned = ms.tir_integration.compile_tir(database, MyMatMulModule, "nvidia/nvidia-a30")

sch_tuned.mod.show()


from tvm.script.parser.tir import evaluate
num_flop = 2 * M * N * K
rt_mod = tvm.build(sch_tuned.mod, target="nvidia/nvidia-a30")
dev = tvm.cuda(0)
A_np = np.random.uniform(size=(M, K)).astype("float32")
B_np = np.random.uniform(size=(K, N)).astype("float32")
A_nd = tvm.nd.array(A_np, dev)
B_nd = tvm.nd.array(B_np, dev)
C_nd = tvm.nd.array(np.zeros((M, N), dtype="float32"), dev)
evaluator = rt_mod.time_evaluator("main", dev, number=10)
print("MetaSchedule: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))


# print(rt_mod.imported_modules[0].get_source())
with open('matmul_tvm_tuned.h', 'w') as f:
  f.write(rt_mod.imported_modules[0].get_source())

