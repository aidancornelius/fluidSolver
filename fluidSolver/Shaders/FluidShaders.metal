//
//  FluidShaders.metal
//  fluidSolver
//
//  Created by Aidan Cornelius-Bell on 6/9/2025.
//  Based on the original ofxMSAFluid library, Copyright (c) 2008-2012 Memo Akten.
//

/// Metal shaders for GPU-accelerated fluid simulation.
///
/// Implements the stable fluids algorithm with enhancements for visual quality.
/// All kernels operate on 2D textures representing fluid fields.

#include <metal_stdlib>
using namespace metal;

/// Converts 2D coordinates to 1D array index.
///
/// Used for compatibility with linear array layouts.
/// - Parameters:
///   - x: X coordinate
///   - y: Y coordinate
///   - width: Grid width
/// - Returns: Linear array index
inline int FLUID_IX(int x, int y, int width) {
    return x + y * width;
}

/// Adds external forces and colour to the fluid simulation.
///
/// Applies Gaussian-weighted force and colour at a specified position.
/// The influence decreases with distance from the centre point.
///
/// - Parameters:
///   - velocity: Velocity field texture (read-write)
///   - density: Density/colour field texture (read-write)
///   - position: Centre position for force application
///   - force: Force vector to apply
///   - color: RGBA colour to add
///   - radius: Influence radius in pixels
///   - deltaTime: Time step for scaling
///   - gid: Thread position in grid
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

/// Performs diffusion step using Jacobi iteration.
///
/// Spreads momentum (for velocity) or colour (for density) to neighbouring cells.
/// Uses implicit solver for numerical stability.
///
/// - Parameters:
///   - input: Source field texture
///   - output: Destination field texture
///   - viscosity: Diffusion coefficient
///   - deltaTime: Time step
///   - gid: Thread position in grid
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

/// Performs semi-Lagrangian advection.
///
/// Transports field values along velocity streamlines.
/// Uses backward particle tracing with bilinear interpolation.
///
/// - Parameters:
///   - velocity: Velocity field for transport
///   - input: Field to advect
///   - output: Advected field result
///   - deltaTime: Time step
///   - gid: Thread position in grid
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

/// Calculates velocity field divergence.
///
/// Measures local expansion/compression for pressure projection.
/// Used to enforce incompressibility constraint.
///
/// - Parameters:
///   - velocity: Velocity field
///   - divergence: Output divergence field
///   - gid: Thread position in grid
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

/// Solves Poisson equation for pressure.
///
/// Iterative solver that finds pressure field to make velocity
/// field divergence-free (incompressible).
///
/// - Parameters:
///   - divergence: Velocity divergence field
///   - pressure: Current pressure estimate
///   - pressureOut: Updated pressure estimate
///   - gid: Thread position in grid
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

/// Subtracts pressure gradient from velocity.
///
/// Final projection step that removes divergent component
/// of velocity field to enforce incompressibility.
///
/// - Parameters:
///   - pressure: Computed pressure field
///   - velocity: Velocity field to correct (read-write)
///   - gid: Thread position in grid
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

/// Calculates vorticity (curl) of velocity field.
///
/// Measures local rotation intensity for vorticity confinement.
/// Helps preserve small-scale turbulent features.
///
/// - Parameters:
///   - velocity: Velocity field
///   - vorticity: Output vorticity field
///   - gid: Thread position in grid
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

/// Applies vorticity confinement forces.
///
/// Adds rotational forces to enhance swirling motion and
/// counteract numerical dissipation.
///
/// - Parameters:
///   - vorticity: Vorticity field
///   - velocity: Velocity field to modify (read-write)
///   - strength: Confinement strength
///   - deltaTime: Time step
///   - gid: Thread position in grid
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

/// Fades density/colour over time.
///
/// Creates dissipation effect and prevents permanent accumulation.
///
/// - Parameters:
///   - density: Density field to fade (read-write)
///   - fadeRate: Fade amount per frame (0-1)
///   - gid: Thread position in grid
kernel void fadeDensity(texture2d<float, access::read_write> density [[texture(0)]],
                       constant float &fadeRate [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    
    float4 dens = density.read(gid);
    dens *= (1.0 - fadeRate);
    dens = max(dens, 0.0);
    density.write(dens, gid);
}

/// Clears texture to zero.
///
/// Used for initialisation and reset operations.
///
/// - Parameters:
///   - texture: Texture to clear (write-only)
///   - gid: Thread position in grid
kernel void clearTexture(texture2d<float, access::write> texture [[texture(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    texture.write(float4(0), gid);
}

/// Renders density field for display.
///
/// Converts density values to visible colours with brightness adjustment.
///
/// - Parameters:
///   - density: Density field to render
///   - output: Display texture
///   - brightness: Brightness multiplier
///   - gid: Thread position in grid
kernel void renderDensity(texture2d<float, access::read> density [[texture(0)]],
                         texture2d<float, access::write> output [[texture(1)]],
                         constant float &brightness [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    
    float4 dens = density.read(gid);
    float4 color = dens * brightness;
    color.a = 1.0;
    output.write(color, gid);
}

/// Renders velocity field as colour-coded motion.
///
/// Maps velocity components to RGB channels for visualisation.
/// Red shows horizontal motion, green shows vertical, blue shows speed.
///
/// - Parameters:
///   - velocity: Velocity field to render
///   - output: Display texture
///   - scale: Velocity scaling factor
///   - gid: Thread position in grid
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

/// Renders velocity magnitude as greyscale.
///
/// Visualises flow speed intensity regardless of direction.
///
/// - Parameters:
///   - velocity: Velocity field
///   - output: Display texture
///   - scale: Speed scaling factor
///   - gid: Thread position in grid
kernel void renderSpeed(texture2d<float, access::read> velocity [[texture(0)]],
                       texture2d<float, access::write> output [[texture(1)]],
                       constant float &scale [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    
    float2 vel = velocity.read(gid).xy;
    float speed = length(vel) * scale;
    float3 color = float3(speed);
    output.write(float4(color, 1.0), gid);
}

/// Particle data structure for GPU processing.
///
/// Matches Swift-side Particle struct for data transfer.
struct Particle {
    float2 position;
    float2 velocity;
    float4 color;
    float life;
    float mass;
    float age;
};

/// Vertex shader output for particle rendering.
struct ParticleOutput {
    float4 position [[position]];
    float4 color;
    float pointSize [[point_size]];
};

/// Vertex shader for particle rendering.
///
/// Generates line primitives from particle positions and velocities
/// to create motion trail effects.
///
/// - Parameters:
///   - vertexID: Vertex identifier (2 per particle for lines)
///   - particles: Particle data buffer
///   - viewportSize: Screen dimensions for coordinate conversion
///   - particleScale: Visual scale factor for trails
/// - Returns: Transformed vertex data
vertex ParticleOutput particleVertex(uint vertexID [[vertex_id]],
                                    constant Particle *particles [[buffer(0)]],
                                    constant float2 &viewportSize [[buffer(1)]],
                                    constant float &particleScale [[buffer(2)]]) {
    ParticleOutput out;
    
    // Each particle generates 2 vertices for a line (like original)
    uint particleIndex = vertexID / 2;
    bool isStart = (vertexID % 2) == 0;
    
    Particle particle = particles[particleIndex];
    
    // Line from (pos - vel) to pos, like original
    // Scale the velocity trail based on particle size
    float2 pos;
    if (isStart) {
        pos = particle.position - particle.velocity * particleScale;  // Trail start, scaled
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

/// Fragment shader for particle rendering.
///
/// Passes through particle colour with alpha blending.
///
/// - Parameter in: Interpolated vertex data
/// - Returns: Final pixel colour
fragment float4 particleFragment(ParticleOutput in [[stage_in]]) {
    return in.color;
}
