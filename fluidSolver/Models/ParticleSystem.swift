//
//  ParticleSystem.swift
//  fluidSolver
//
//  Created by Aidan Cornelius-Bell on 6/9/2025.
//  Based on the original ofxMSAFluid library, Copyright (c) 2008-2012 Memo Akten.
//

import Foundation
import Metal
import MetalKit
import simd

/// Represents a single particle in the fluid simulation.
///
/// Particles follow the fluid velocity field initially then coast on momentum,
/// creating visual trails that enhance the fluid motion.
struct Particle {
    /// Current position in screen coordinates
    var position: SIMD2<Float>
    /// Current velocity vector
    var velocity: SIMD2<Float>
    /// RGBA colour with alpha for transparency
    var color: SIMD4<Float>
    /// Remaining life (0-1), controls opacity and removal
    var life: Float
    /// Particle mass affecting fluid interaction strength
    var mass: Float
    /// Time since spawn in seconds, affects fluid coupling
    var age: Float = 0
}

/// Manages a system of particles that interact with the fluid simulation.
///
/// Particles are spawned at interaction points and follow the fluid velocity
/// field with decreasing influence over time. They provide visual feedback
/// for fluid motion through additive blending and motion trails.
class ParticleSystem: ObservableObject {
    // MARK: - Properties
    
    /// Metal device for GPU operations
    private var device: MTLDevice!
    /// Active particles in the system
    private var particles: [Particle] = []
    /// GPU buffer for particle data
    private var particleBuffer: MTLBuffer?
    /// Unused vertex buffer (kept for compatibility)
    private var vertexBuffer: MTLBuffer?
    /// Render pipeline for particle drawing
    private var renderPipelineState: MTLRenderPipelineState!
    
    /// Maximum number of particles allowed
    private let maxParticles = 50000
    /// Current insertion index for cycling (unused)
    private var currentIndex = 0
    
    // MARK: - Physics parameters
    
    /// Momentum preservation factor.
    ///
    /// Controls how much velocity particles retain over time.
    /// Range: 0.0 to 1.0
    @Published var momentum: Float = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.momentum.rawValue) != nil ? UserDefaults.standard.float(forKey: UserDefaults.FluidKeys.momentum.rawValue) : 0.5 {
        didSet { UserDefaults.standard.set(momentum, forKey: UserDefaults.FluidKeys.momentum.rawValue) }
    }
    /// Strength of fluid velocity influence on particles.
    ///
    /// Higher values make particles follow fluid more closely.
    /// Range: 0.0 to 1.0
    @Published var fluidForce: Float = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.fluidForce.rawValue) != nil ? UserDefaults.standard.float(forKey: UserDefaults.FluidKeys.fluidForce.rawValue) : 0.6 {
        didSet { UserDefaults.standard.set(fluidForce, forKey: UserDefaults.FluidKeys.fluidForce.rawValue) }
    }
    /// Rate at which particles fade out.
    ///
    /// Controls particle lifetime and trail length.
    /// Range: 0.0 to 0.01
    @Published var fadeSpeed: Float = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.fadeSpeedParticles.rawValue) != nil ? UserDefaults.standard.float(forKey: UserDefaults.FluidKeys.fadeSpeedParticles.rawValue) : 0.001 {
        didSet { UserDefaults.standard.set(fadeSpeed, forKey: UserDefaults.FluidKeys.fadeSpeedParticles.rawValue) }
    }
    /// Number of particles spawned per interaction.
    ///
    /// Higher values create denser particle clouds.
    /// Range: 1 to 50
    @Published var spawnCount: Int = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.spawnCount.rawValue) != nil ? UserDefaults.standard.integer(forKey: UserDefaults.FluidKeys.spawnCount.rawValue) : 10 {
        didSet { UserDefaults.standard.set(spawnCount, forKey: UserDefaults.FluidKeys.spawnCount.rawValue) }
    }
    /// Radius for random particle spawn positions.
    ///
    /// Creates spread around interaction point.
    /// Range: 0.0 to 30.0 pixels
    @Published var spawnRadius: Float = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.spawnRadius.rawValue) != nil ? UserDefaults.standard.float(forKey: UserDefaults.FluidKeys.spawnRadius.rawValue) : 15.0 {
        didSet { UserDefaults.standard.set(spawnRadius, forKey: UserDefaults.FluidKeys.spawnRadius.rawValue) }
    }
    /// Rate at which fluid influence decreases with particle age.
    ///
    /// Higher values make particles break away from fluid sooner.
    /// Range: 0.5 to 10.0
    @Published var fluidDecayRate: Float = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.fluidDecayRate.rawValue) != nil ? UserDefaults.standard.float(forKey: UserDefaults.FluidKeys.fluidDecayRate.rawValue) : 3.0 {
        didSet { UserDefaults.standard.set(fluidDecayRate, forKey: UserDefaults.FluidKeys.fluidDecayRate.rawValue) }
    }
    /// Links particle fluid coupling to fluid viscosity.
    ///
    /// When enabled, more viscous fluids affect particles more strongly.
    @Published var linkToViscosity: Bool = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.linkToViscosity.rawValue) != nil ? UserDefaults.standard.bool(forKey: UserDefaults.FluidKeys.linkToViscosity.rawValue) : true {
        didSet { UserDefaults.standard.set(linkToViscosity, forKey: UserDefaults.FluidKeys.linkToViscosity.rawValue) }
    }
    /// Visual size of particles.
    ///
    /// Affects rendering scale but not physics.
    /// Range: 0.1 to 5.0
    @Published var particleSize: Float = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.particleSize.rawValue) != nil ? UserDefaults.standard.float(forKey: UserDefaults.FluidKeys.particleSize.rawValue) : 1.0 {
        didSet { UserDefaults.standard.set(particleSize, forKey: UserDefaults.FluidKeys.particleSize.rawValue) }
    }
    
    /// Current number of active particles
    @Published var particleCount: Int = 0
    /// Master enable/disable for particle system
    @Published var isEnabled: Bool = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.particlesEnabled.rawValue) != nil ? UserDefaults.standard.bool(forKey: UserDefaults.FluidKeys.particlesEnabled.rawValue) : true {
        didSet { UserDefaults.standard.set(isEnabled, forKey: UserDefaults.FluidKeys.particlesEnabled.rawValue) }
    }
    
    /// Initialises the particle system with Metal device.
    ///
    /// Sets up render pipeline and allocates GPU buffers.
    ///
    /// - Parameter device: Metal device for GPU operations
    init(device: MTLDevice) {
        self.device = device
        setupPipeline()
        setupBuffers()
    }
    
    /// Configures the Metal render pipeline for particles.
    ///
    /// Sets up additive blending for glow effect and loads
    /// particle vertex and fragment shaders.
    private func setupPipeline() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Failed to load Metal library")
        }
        
        let vertexFunction = library.makeFunction(name: "particleVertex")
        let fragmentFunction = library.makeFunction(name: "particleFragment")
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        // Enable blending for particles (additive for glow effect)
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .one  // Additive blending
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .one
        
        // No vertex descriptor needed since we're using vertex ID indexing
        
        do {
            renderPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            fatalError("Failed to create particle render pipeline: \(error)")
        }
    }
    
    /// Allocates GPU buffer for particle data.
    private func setupBuffers() {
        let bufferSize = MemoryLayout<Particle>.stride * maxParticles
        particleBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
    }
    
    /// Previous interaction position (unused)
    private var previousPosition: CGPoint?
    
    /// Spawns new particles at the specified position.
    ///
    /// Particles are created with random offsets and initial properties,
    /// starting with zero velocity to be influenced by fluid field.
    ///
    /// - Parameters:
    ///   - position: Spawn position in screen coordinates
    ///   - count: Number of particles to spawn
    ///   - windowSize: Current window dimensions
    func addParticles(at position: CGPoint, count: Int = 10, windowSize: CGSize) {
        guard isEnabled else { return }
        
        // Use window coordinates directly (particles render in screen space)
        let basePos = SIMD2<Float>(
            Float(position.x),
            Float(position.y)
        )
        
        for _ in 0..<count {
            if particles.count >= maxParticles {
                // Cycle through like original (curIndex wrapping)
                particles.removeFirst()
            }
            
            // Match ofxMSAFluid: pos + randVec2f() * 15
            let randomOffset = SIMD2<Float>(
                Float.random(in: -spawnRadius...spawnRadius),
                Float.random(in: -spawnRadius...spawnRadius)
            )
            
            // Original starts with zero velocity
            let initialVel = SIMD2<Float>(0, 0)
            
            // Match original alpha and mass ranges
            let alpha = Float.random(in: 0.3...1.0)  // Original: 0.3 to 1
            let mass = Float.random(in: 0.1...1.0)   // Original: 0.1 to 1
            
            // White particles
            let color = SIMD4<Float>(1.0, 1.0, 1.0, alpha)
            
            let particle = Particle(
                position: basePos + randomOffset,
                velocity: initialVel,
                color: color,
                life: alpha,  // Use alpha as life
                mass: mass,
                age: 0  // New particle starts at age 0
            )
            
            particles.append(particle)
        }
        
        particleCount = particles.count
        updateBuffer()
    }
    
    /// Updates particle physics and removes dead particles.
    ///
    /// Particles follow fluid velocity with decreasing influence over time,
    /// transitioning to momentum-based motion. Removes particles that
    /// leave the screen or fade below threshold.
    ///
    /// - Parameters:
    ///   - fluidSolver: Fluid solver for velocity queries
    ///   - windowSize: Current window dimensions
    func update(fluidSolver: FluidSolver, windowSize: CGSize) {
        guard isEnabled, !particles.isEmpty else { return }
        
        let invWindowSize = SIMD2<Float>(
            1.0 / Float(windowSize.width),
            1.0 / Float(windowSize.height)
        )
        
        // Update each particle
        for i in particles.indices.reversed() {
            var particle = particles[i]
            
            // Particles follow fluid strongly at first, then coast on momentum
            let fluidPos = particle.position * invWindowSize
            let fluidVel = fluidSolver.getVelocityAt(normalizedPos: fluidPos)
            
            // Reduce fluid influence rapidly after spawn
            let ageDecay = max(0, 1.0 - particle.age * fluidDecayRate)
            
            // Link fluid force to viscosity if enabled (more viscous = particles follow more)
            let effectiveFluidForce: Float
            if linkToViscosity {
                // Map viscosity (0...0.01) to reasonable fluid force (0.1...1.0)
                effectiveFluidForce = 0.1 + (fluidSolver.viscosity / 0.01) * 0.9
            } else {
                effectiveFluidForce = fluidForce
            }
            
            // Apply fluid force with decay, increase momentum as fluid influence decreases
            let fluidContribution = fluidVel * (particle.mass * effectiveFluidForce * ageDecay) * SIMD2<Float>(Float(windowSize.width), Float(windowSize.height))
            let effectiveMomentum = momentum + (1.0 - momentum) * min(1.0, particle.age * 2.0)  // Momentum increases as particle ages
            let momentumContribution = particle.velocity * effectiveMomentum
            
            particle.velocity = fluidContribution + momentumContribution
            particle.position += particle.velocity
            
            // Age the particle (assuming 60fps)
            particle.age += 1.0 / 60.0
            
            // Remove particles that go off-screen
            let margin: Float = 50  // Small margin to ensure they're fully off-screen
            if particle.position.x < -margin || 
               particle.position.x > Float(windowSize.width) + margin ||
               particle.position.y < -margin || 
               particle.position.y > Float(windowSize.height) + margin {
                particle.life = 0  // Mark for removal
            }
            
            // Original fade: alpha *= 0.999f
            particle.life *= 0.999
            
            // Original kills particle if alpha < 0.01
            if particle.life < 0.01 {
                particle.life = 0
                particles.remove(at: i)
            } else {
                particles[i] = particle
            }
        }
        
        particleCount = particles.count
        updateBuffer()
    }
    
    /// Samples fluid velocity at a position.
    ///
    /// Currently returns zero as actual sampling is done through
    /// the fluid solver's getVelocityAt method.
    ///
    /// - Parameters:
    ///   - position: Position to sample
    ///   - solver: Fluid solver reference
    /// - Returns: Velocity vector (currently zero)
    private func sampleFluidVelocity(at position: SIMD2<Float>, solver: FluidSolver) -> SIMD2<Float> {
        // Sample from the actual fluid solver's velocity field
        // Note: This is a simplified sampling - ideally we'd read directly from the texture
        // For now, we'll return a velocity that creates proper fluid-following behavior
        
        // The fluid solver expects normalized coordinates (0-1)
        // and the particles should follow the fluid velocity closely
        // In the original, particles query: solver.getVelocityAtPos(pos * invWindowSize)
        
        // Return a small base velocity - the real velocity comes from the fluid solver's internal state
        // This creates the clustering/following behavior seen in the original
        return SIMD2<Float>(0, 0)  // Will be replaced by actual fluid force
    }
    
    /// Updates GPU buffer with current particle data.
    ///
    /// Copies particle array to GPU memory for rendering.
    private func updateBuffer() {
        guard let buffer = particleBuffer else { return }
        
        let pointer = buffer.contents().bindMemory(to: Particle.self, capacity: particles.count)
        for (index, particle) in particles.enumerated() {
            pointer[index] = particle
        }
    }
    
    /// Renders particles using Metal.
    ///
    /// Draws particles as lines to create motion trail effect
    /// with additive blending for glow.
    ///
    /// - Parameters:
    ///   - renderEncoder: Active render command encoder
    ///   - viewportSize: Current viewport dimensions
    func render(in renderEncoder: MTLRenderCommandEncoder, viewportSize: CGSize) {
        guard isEnabled, !particles.isEmpty, let buffer = particleBuffer else { return }
        
        renderEncoder.setRenderPipelineState(renderPipelineState)
        renderEncoder.setVertexBuffer(buffer, offset: 0, index: 0)
        
        var viewport = SIMD2<Float>(Float(viewportSize.width), Float(viewportSize.height))
        renderEncoder.setVertexBytes(&viewport, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
        
        var scale = particleSize
        renderEncoder.setVertexBytes(&scale, length: MemoryLayout<Float>.size, index: 2)
        
        // Draw as lines to show motion trails (like original)
        renderEncoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: particles.count * 2)
    }
    
    /// Clears all particles from the system.
    func reset() {
        particles.removeAll()
        particleCount = 0
        updateBuffer()
    }
    
    /// Converts HSB colour to RGBA.
    ///
    /// Used for generating colourful particles based on time or position.
    ///
    /// - Parameters:
    ///   - h: Hue (0-1)
    ///   - s: Saturation (0-1)
    ///   - b: Brightness (0-1)
    /// - Returns: RGBA colour vector
    private func hsbToRgb(h: Float, s: Float, b: Float) -> SIMD4<Float> {
        let c = b * s
        let x = c * (1 - abs(fmod(h * 6, 2) - 1))
        let m = b - c
        
        var rgb: SIMD3<Float>
        if h < 1.0/6.0 {
            rgb = SIMD3(c, x, 0)
        } else if h < 2.0/6.0 {
            rgb = SIMD3(x, c, 0)
        } else if h < 3.0/6.0 {
            rgb = SIMD3(0, c, x)
        } else if h < 4.0/6.0 {
            rgb = SIMD3(0, x, c)
        } else if h < 5.0/6.0 {
            rgb = SIMD3(x, 0, c)
        } else {
            rgb = SIMD3(c, 0, x)
        }
        
        return SIMD4(rgb + SIMD3(repeating: m), 1.0)
    }
}
