// where: y = cond > 0.5 ? a : b.
// Using vectorized loads/stores via float4 for better memory bandwidth utilization.
// Optimized: increased vector width to float8 for fewer memory transactions.
// Attempt: use simdgroup operations to reduce global memory traffic by having
// each simdgroup load a tile and share results via threadgroup memory.
// New approach: reduce grid size and increase per-thread work to 16 elements
// (4 float4 loads) to amortize dispatch overhead and improve cache reuse.
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
kernel void where_f32(
    device const float*  cond      [[buffer(0)]],
    device const float*  a         [[buffer(1)]],
    device const float*  b         [[buffer(2)]],
    device       float*  y         [[buffer(3)]],
    constant     uint&   N         [[buffer(4)]],
    constant     uint&   grid_size [[buffer(5)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const uint gs = grid_size;
    uint i = tid * 16;
    
    // Process 16 elements per thread using 4 float4 loads for coalesced access
    for (; i + 15 < N; i += gs * 16) {
        // Load 16 floats as four float4s
        float4 c1 = *((device float4*)(cond + i));
        float4 c2 = *((device float4*)(cond + i + 4));
        float4 c3 = *((device float4*)(cond + i + 8));
        float4 c4 = *((device float4*)(cond + i + 12));
        
        float4 av1 = *((device float4*)(a + i));
        float4 av2 = *((device float4*)(a + i + 4));
        float4 av3 = *((device float4*)(a + i + 8));
        float4 av4 = *((device float4*)(a + i + 12));
        
        float4 bv1 = *((device float4*)(b + i));
        float4 bv2 = *((device float4*)(b + i + 4));
        float4 bv3 = *((device float4*)(b + i + 8));
        float4 bv4 = *((device float4*)(b + i + 12));
        
        // Use Metal's native select with bool4 masks
        bool4 mask1 = bool4(c1.x > 0.5f, c1.y > 0.5f, c1.z > 0.5f, c1.w > 0.5f);
        bool4 mask2 = bool4(c2.x > 0.5f, c2.y > 0.5f, c2.z > 0.5f, c2.w > 0.5f);
        bool4 mask3 = bool4(c3.x > 0.5f, c3.y > 0.5f, c3.z > 0.5f, c3.w > 0.5f);
        bool4 mask4 = bool4(c4.x > 0.5f, c4.y > 0.5f, c4.z > 0.5f, c4.w > 0.5f);
        
        float4 result1 = select(bv1, av1, mask1);
        float4 result2 = select(bv2, av2, mask2);
        float4 result3 = select(bv3, av3, mask3);
        float4 result4 = select(bv4, av4, mask4);
        
        *((device float4*)(y + i)) = result1;
        *((device float4*)(y + i + 4)) = result2;
        *((device float4*)(y + i + 8)) = result3;
        *((device float4*)(y + i + 12)) = result4;
    }
    
    // Remainder loop (scalar)
    for (; i < N; i += gs) {
        y[i] = cond[i] > 0.5f ? a[i] : b[i];
    }
}
