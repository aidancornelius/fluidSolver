import Foundation
import Metal
import MetalKit
import simd

struct Particle {
    var position: SIMD2<Float>
    var velocity: SIMD2<Float>
    var color: SIMD4<Float>
    var life: Float
    var mass: Float
    var age: Float = 0  // Track how long particle has existed
}

class ParticleSystem: ObservableObject {
    private var device: MTLDevice!
    private var particles: [Particle] = []
    private var particleBuffer: MTLBuffer?
    private var vertexBuffer: MTLBuffer?
    private var renderPipelineState: MTLRenderPipelineState!
    
    private let maxParticles = 50000
    private var currentIndex = 0
    
    // Physics constants matching ofxMSAFluid exactly
    @Published var momentum: Float = 0.5  // MOMENTUM in original
    @Published var fluidForce: Float = 0.6  // FLUID_FORCE in original
    @Published var fadeSpeed: Float = 0.001  // Alpha *= 0.999 in original
    @Published var spawnCount: Int = 10  // Particles per spawn
    @Published var spawnRadius: Float = 15.0  // Original uses 15 pixel radius
    @Published var fluidDecayRate: Float = 3.0  // How quickly particles stop following fluid (higher = faster)
    
    @Published var particleCount: Int = 0
    @Published var isEnabled: Bool = true
    
    init(device: MTLDevice) {
        self.device = device
        setupPipeline()
        setupBuffers()
    }
    
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
    
    private func setupBuffers() {
        let bufferSize = MemoryLayout<Particle>.stride * maxParticles
        particleBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
    }
    
    private var previousPosition: CGPoint?
    
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
            
            // Apply fluid force with decay, increase momentum as fluid influence decreases
            let fluidContribution = fluidVel * (particle.mass * fluidForce * ageDecay) * SIMD2<Float>(Float(windowSize.width), Float(windowSize.height))
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
    
    private func updateBuffer() {
        guard let buffer = particleBuffer else { return }
        
        let pointer = buffer.contents().bindMemory(to: Particle.self, capacity: particles.count)
        for (index, particle) in particles.enumerated() {
            pointer[index] = particle
        }
    }
    
    func render(in renderEncoder: MTLRenderCommandEncoder, viewportSize: CGSize) {
        guard isEnabled, !particles.isEmpty, let buffer = particleBuffer else { return }
        
        renderEncoder.setRenderPipelineState(renderPipelineState)
        renderEncoder.setVertexBuffer(buffer, offset: 0, index: 0)
        
        var viewport = SIMD2<Float>(Float(viewportSize.width), Float(viewportSize.height))
        renderEncoder.setVertexBytes(&viewport, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
        
        // Draw as lines to show motion trails (like original)
        renderEncoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: particles.count * 2)
    }
    
    func reset() {
        particles.removeAll()
        particleCount = 0
        updateBuffer()
    }
    
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