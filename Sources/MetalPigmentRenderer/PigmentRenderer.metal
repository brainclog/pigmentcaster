#include <metal_stdlib>
using namespace metal;

struct Params {
    uint  width;
    uint  height;
    uint  bagSize;
    uint  nPhotons;
    uint  nWavelengths;
    uint  nPigments;
    uint  seed;
};

// Simple xorshift-like RNG per thread
struct RNG {
    uint state;
    inline uint next_u32() thread {
        uint x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }
    inline float next_f32() thread {
        // use lower 24 bits for a 0..1 float
        return float(next_u32() & 0x00FFFFFF) / 16777216.0;
    }
    inline uint next_range(uint n) thread {
        return next_u32() % n;
    }
};

// Kernel:
// - pigmentTable: constant float [nPigments * nWavelengths]
// - canvasBags: constant uchar [width * height * bagSize] (pigment IDs per slot)
// - outTex: texture2d<float, access::write> (rgba32Float)
kernel void render_canvas_kernel(
    constant Params&            p                 [[ buffer(0) ]],
    constant float*   __restrict pigmentTable     [[ buffer(1) ]],
    constant uchar*   __restrict canvasBags       [[ buffer(2) ]],
    texture2d<float, access::write> outTex        [[ texture(0) ]],
    uint2 gid                                    [[ thread_position_in_grid ]]
) {
    if (gid.x >= p.width || gid.y >= p.height) { return; }

    uint idx = gid.y * p.width + gid.x;

    // seed RNG from pixel index + global seed
    RNG rng;
    uint s = idx ^ (p.seed * 747796405u + 2891336453u);
    s ^= (s >> 16);
    rng.state = s;

    // pointer into the canvas bag for this pixel (device-side buffer is 'constant' here)
    const constant uchar* pixelBag = canvasBags + idx * p.bagSize;

    // accumulate survived counts for 6 spectral bins
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f, s5 = 0.0f;

    for (uint n = 0; n < p.nPhotons; ++n) {
        uint wl = rng.next_range(p.nWavelengths);   // 0..5
        uint slot = rng.next_range(p.bagSize);      // 0..bagSize-1
        uint pid = (uint) pixelBag[slot];           // pigment id index

        // index into pigment table: pid * nWavelengths + wl
        float absorb = pigmentTable[pid * p.nWavelengths + wl];
        float survive = (rng.next_f32() > absorb) ? 1.0f : 0.0f;

        // accumulate to the corresponding spectral bin
        switch (wl) {
            case 0u: s0 += survive; break;
            case 1u: s1 += survive; break;
            case 2u: s2 += survive; break;
            case 3u: s3 += survive; break;
            case 4u: s4 += survive; break;
            case 5u: s5 += survive; break;
        }
    }

    float invN = 1.0f / float(p.nPhotons);
    float R = clamp((s0 + s1) * invN, 0.0f, 1.0f);
    float G = clamp((s2 + s3) * invN, 0.0f, 1.0f);
    float B = clamp((s4 + s5) * invN, 0.0f, 1.0f);

    float brightness = 1.5;
    R *= brightness;
    G *= brightness;
    B *= brightness;

    // clamp before gamma
    R = min(R * brightness, 1.0);
    G = min(G * brightness, 1.0);
    B = min(B * brightness, 1.0);

    // apply gamma 2.2
    float gamma = 2.8;
    float4 outc = float4(pow(R, 1.0/gamma),
                         pow(G, 1.0/gamma),
                         pow(B, 1.0/gamma),
                         1.0);
    outTex.write(outc, gid);
}
