#include <metal_stdlib>
using namespace metal;

// Inline function for grid indexing
inline int FLUID_IX(int x, int y, int width) {
    return x + y * width;
}

// Kernel for adding forces and dye to the fluid
kernel void addForces(texture2d<float, access::read_write> velocity [[texture(0)]],
                      texture2d<float, access::read_write> density [[texture(1)]],
                      constant float2 &position [[buffer(0)]],
                      constant float2 &force [[buffer(1)]],
                      constant float4 &color [[buffer(2)]],
                      constant float &radius [[buffer(3)]],
                      constant float &deltaTime [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    
    float2 pos = float2(gid);
    float2 diff = pos - position;
    float dist = length(diff);
    
    if (dist < radius) {
        float influence = exp(-dist * dist / (2.0 * radius * radius));
        
        // Add force to velocity
        float4 vel = velocity.read(gid);
        vel.xy += force * influence * deltaTime;
        velocity.write(vel, gid);
        
        // Add dye to density
        float4 dens = density.read(gid);
        dens += color * influence * deltaTime;
        density.write(dens, gid);
    }
}

// Diffusion kernel using Jacobi iteration
kernel void diffuse(texture2d<float, access::read> input [[texture(0)]],
                   texture2d<float, access::write> output [[texture(1)]],
                   constant float &viscosity [[buffer(0)]],
                   constant float &deltaTime [[buffer(1)]],
                   uint2 gid [[thread_position_in_grid]]) {
    
    uint width = input.get_width();
    uint height = input.get_height();
    
    if (gid.x == 0 || gid.x >= width - 1 || gid.y == 0 || gid.y >= height - 1) {
        output.write(input.read(gid), gid);
        return;
    }
    
    float a = deltaTime * viscosity * width * height;
    float4 center = input.read(gid);
    float4 left = input.read(uint2(gid.x - 1, gid.y));
    float4 right = input.read(uint2(gid.x + 1, gid.y));
    float4 top = input.read(uint2(gid.x, gid.y - 1));
    float4 bottom = input.read(uint2(gid.x, gid.y + 1));
    
    float4 result = (center + a * (left + right + top + bottom)) / (1.0 + 4.0 * a);
    output.write(result, gid);
}

// Semi-Lagrangian advection
kernel void advect(texture2d<float, access::read> velocity [[texture(0)]],
                  texture2d<float, access::read> input [[texture(1)]],
                  texture2d<float, access::write> output [[texture(2)]],
                  constant float &deltaTime [[buffer(0)]],
                  uint2 gid [[thread_position_in_grid]]) {
    
    uint width = input.get_width();
    uint height = input.get_height();
    
    if (gid.x == 0 || gid.x >= width - 1 || gid.y == 0 || gid.y >= height - 1) {
        output.write(input.read(gid), gid);
        return;
    }
    
    float2 vel = velocity.read(gid).xy;
    float2 pos = float2(gid) - vel * deltaTime;
    
    // Clamp to boundaries
    pos.x = clamp(pos.x, 0.5f, width - 1.5f);
    pos.y = clamp(pos.y, 0.5f, height - 1.5f);
    
    // Bilinear interpolation
    int2 i0 = int2(floor(pos));
    int2 i1 = i0 + 1;
    float2 s = pos - float2(i0);
    
    float4 q00 = input.read(uint2(i0.x, i0.y));
    float4 q10 = input.read(uint2(i1.x, i0.y));
    float4 q01 = input.read(uint2(i0.x, i1.y));
    float4 q11 = input.read(uint2(i1.x, i1.y));
    
    float4 result = mix(mix(q00, q10, s.x), mix(q01, q11, s.x), s.y);
    output.write(result, gid);
}

// Calculate divergence for projection step
kernel void divergence(texture2d<float, access::read> velocity [[texture(0)]],
                      texture2d<float, access::write> divergence [[texture(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
    
    uint width = velocity.get_width();
    uint height = velocity.get_height();
    
    if (gid.x == 0 || gid.x >= width - 1 || gid.y == 0 || gid.y >= height - 1) {
        divergence.write(float4(0), gid);
        return;
    }
    
    float2 left = velocity.read(uint2(gid.x - 1, gid.y)).xy;
    float2 right = velocity.read(uint2(gid.x + 1, gid.y)).xy;
    float2 top = velocity.read(uint2(gid.x, gid.y - 1)).xy;
    float2 bottom = velocity.read(uint2(gid.x, gid.y + 1)).xy;
    
    float div = -0.5 * ((right.x - left.x) + (bottom.y - top.y));
    divergence.write(float4(div, 0, 0, 0), gid);
}

// Pressure solve using Jacobi iteration
kernel void pressureSolve(texture2d<float, access::read> divergence [[texture(0)]],
                         texture2d<float, access::read> pressure [[texture(1)]],
                         texture2d<float, access::write> pressureOut [[texture(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    
    uint width = pressure.get_width();
    uint height = pressure.get_height();
    
    if (gid.x == 0 || gid.x >= width - 1 || gid.y == 0 || gid.y >= height - 1) {
        pressureOut.write(float4(0), gid);
        return;
    }
    
    float div = divergence.read(gid).x;
    float left = pressure.read(uint2(gid.x - 1, gid.y)).x;
    float right = pressure.read(uint2(gid.x + 1, gid.y)).x;
    float top = pressure.read(uint2(gid.x, gid.y - 1)).x;
    float bottom = pressure.read(uint2(gid.x, gid.y + 1)).x;
    
    float p = (div + left + right + top + bottom) / 4.0;
    pressureOut.write(float4(p, 0, 0, 0), gid);
}

// Subtract pressure gradient from velocity
kernel void subtractGradient(texture2d<float, access::read> pressure [[texture(0)]],
                            texture2d<float, access::read_write> velocity [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
    
    uint width = velocity.get_width();
    uint height = velocity.get_height();
    
    if (gid.x == 0 || gid.x >= width - 1 || gid.y == 0 || gid.y >= height - 1) {
        return;
    }
    
    float left = pressure.read(uint2(gid.x - 1, gid.y)).x;
    float right = pressure.read(uint2(gid.x + 1, gid.y)).x;
    float top = pressure.read(uint2(gid.x, gid.y - 1)).x;
    float bottom = pressure.read(uint2(gid.x, gid.y + 1)).x;
    
    float4 vel = velocity.read(gid);
    vel.x -= 0.5 * (right - left);
    vel.y -= 0.5 * (bottom - top);
    velocity.write(vel, gid);
}

// Calculate vorticity (curl)
kernel void calculateVorticity(texture2d<float, access::read> velocity [[texture(0)]],
                              texture2d<float, access::write> vorticity [[texture(1)]],
                              uint2 gid [[thread_position_in_grid]]) {
    
    uint width = velocity.get_width();
    uint height = velocity.get_height();
    
    if (gid.x == 0 || gid.x >= width - 1 || gid.y == 0 || gid.y >= height - 1) {
        vorticity.write(float4(0), gid);
        return;
    }
    
    float2 left = velocity.read(uint2(gid.x - 1, gid.y)).xy;
    float2 right = velocity.read(uint2(gid.x + 1, gid.y)).xy;
    float2 top = velocity.read(uint2(gid.x, gid.y - 1)).xy;
    float2 bottom = velocity.read(uint2(gid.x, gid.y + 1)).xy;
    
    float curl = (bottom.x - top.x) - (right.y - left.y);
    vorticity.write(float4(curl, 0, 0, 0), gid);
}

// Apply vorticity confinement force
kernel void applyVorticity(texture2d<float, access::read> vorticity [[texture(0)]],
                          texture2d<float, access::read_write> velocity [[texture(1)]],
                          constant float &strength [[buffer(0)]],
                          constant float &deltaTime [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
    
    uint width = velocity.get_width();
    uint height = velocity.get_height();
    
    if (gid.x == 0 || gid.x >= width - 1 || gid.y == 0 || gid.y >= height - 1) {
        return;
    }
    
    float left = abs(vorticity.read(uint2(gid.x - 1, gid.y)).x);
    float right = abs(vorticity.read(uint2(gid.x + 1, gid.y)).x);
    float top = abs(vorticity.read(uint2(gid.x, gid.y - 1)).x);
    float bottom = abs(vorticity.read(uint2(gid.x, gid.y + 1)).x);
    float center = vorticity.read(gid).x;
    
    float2 gradient = float2(right - left, bottom - top);
    float gradientLength = length(gradient) + 0.00001;
    gradient = gradient / gradientLength;
    
    float2 force = strength * center * float2(gradient.y, -gradient.x);
    
    float4 vel = velocity.read(gid);
    vel.xy += force * deltaTime;
    velocity.write(vel, gid);
}

// Fade density over time
kernel void fadeDensity(texture2d<float, access::read_write> density [[texture(0)]],
                       constant float &fadeRate [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    
    float4 dens = density.read(gid);
    dens *= (1.0 - fadeRate);
    dens = max(dens, 0.0);
    density.write(dens, gid);
}

// Clear texture
kernel void clearTexture(texture2d<float, access::write> texture [[texture(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    texture.write(float4(0), gid);
}

// Render density to display texture
kernel void renderDensity(texture2d<float, access::read> density [[texture(0)]],
                         texture2d<float, access::write> output [[texture(1)]],
                         constant float &brightness [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    
    float4 dens = density.read(gid);
    float4 color = dens * brightness;
    color.a = 1.0;
    output.write(color, gid);
}

// Render velocity field as motion visualization
kernel void renderVelocity(texture2d<float, access::read> velocity [[texture(0)]],
                          texture2d<float, access::write> output [[texture(1)]],
                          constant float &scale [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    
    float2 vel = velocity.read(gid).xy;
    float speed = length(vel);
    
    // Map velocity to color (red = x velocity, green = y velocity, blue = speed)
    // This matches ofxMSAFluid's motion visualization
    float3 color;
    color.r = vel.x * scale + 0.5; // Centered at 0.5
    color.g = vel.y * scale + 0.5; // Centered at 0.5
    color.b = speed * scale;
    
    output.write(float4(color, 1.0), gid);
}

// Render speed as grayscale
kernel void renderSpeed(texture2d<float, access::read> velocity [[texture(0)]],
                       texture2d<float, access::write> output [[texture(1)]],
                       constant float &scale [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    
    float2 vel = velocity.read(gid).xy;
    float speed = length(vel) * scale;
    float3 color = float3(speed);
    output.write(float4(color, 1.0), gid);
}

// Particle struct matching Swift side
struct Particle {
    float2 position;
    float2 velocity;
    float4 color;
    float life;
    float mass;
    float age;
};

struct ParticleOutput {
    float4 position [[position]];
    float4 color;
    float pointSize [[point_size]];
};

vertex ParticleOutput particleVertex(uint vertexID [[vertex_id]],
                                    constant Particle *particles [[buffer(0)]],
                                    constant float2 &viewportSize [[buffer(1)]]) {
    ParticleOutput out;
    
    // Each particle generates 2 vertices for a line (like original)
    uint particleIndex = vertexID / 2;
    bool isStart = (vertexID % 2) == 0;
    
    Particle particle = particles[particleIndex];
    
    // Line from (pos - vel) to pos, like original
    float2 pos;
    if (isStart) {
        pos = particle.position - particle.velocity;  // Trail start
    } else {
        pos = particle.position;  // Trail end
    }
    
    // Convert to NDC
    float2 ndc = (pos / viewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y for Metal coordinate system
    
    out.position = float4(ndc, 0.0, 1.0);
    out.color = float4(1.0, 1.0, 1.0, particle.life);  // White with alpha
    out.pointSize = 1.0;
    
    return out;
}

fragment float4 particleFragment(ParticleOutput in [[stage_in]]) {
    return in.color;
}