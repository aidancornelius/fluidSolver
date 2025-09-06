import Foundation
import Metal
import MetalKit
import simd

class FluidSolver: ObservableObject {
    // Metal objects
    private var device: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var library: MTLLibrary!
    
    // Compute pipeline states
    private var addForcesPipeline: MTLComputePipelineState!
    private var diffusePipeline: MTLComputePipelineState!
    private var advectPipeline: MTLComputePipelineState!
    private var divergencePipeline: MTLComputePipelineState!
    private var pressureSolvePipeline: MTLComputePipelineState!
    private var subtractGradientPipeline: MTLComputePipelineState!
    private var vorticityPipeline: MTLComputePipelineState!
    private var applyVorticityPipeline: MTLComputePipelineState!
    private var fadeDensityPipeline: MTLComputePipelineState!
    private var clearTexturePipeline: MTLComputePipelineState!
    private var renderDensityPipeline: MTLComputePipelineState!
    private var renderVelocityPipeline: MTLComputePipelineState!
    private var renderSpeedPipeline: MTLComputePipelineState!
    
    // Textures
    private var velocityTexture: MTLTexture!
    private var velocityTempTexture: MTLTexture!
    private var densityTexture: MTLTexture!
    private var densityTempTexture: MTLTexture!
    private var pressureTexture: MTLTexture!
    private var pressureTempTexture: MTLTexture!
    private var divergenceTexture: MTLTexture!
    private var vorticityTexture: MTLTexture!
    private var displayTexture: MTLTexture!
    
    // Grid properties
    @Published var gridWidth: Int = 256  // Increased resolution
    @Published var gridHeight: Int = 256  // Increased resolution
    
    // Simulation parameters - matching ofxMSAFluid defaults
    @Published var viscosity: Float = 0.00015  // FLUID_DEFAULT_VISC
    @Published var diffusion: Float = 0.0  // FLUID_DEFAULT_COLOR_DIFFUSION
    @Published var deltaTime: Float = 0.5  // From example settings
    @Published var vorticityStrength: Float = 0.0  // Default off
    @Published var fadeSpeed: Float = 0.002  // FLUID_DEFAULT_FADESPEED
    @Published var solverIterations: Int = 10  // FLUID_DEFAULT_SOLVER_ITERATIONS
    @Published var forceMultiplier: Float = 24.0  // velocityMult from XML
    @Published var colorMultiplier: Float = 100.0  // colorMult from XML
    @Published var brightness: Float = 0.06  // brightness from XML
    @Published var forceRadius: Float = 5.0  // Radius of force application
    
    // Interaction state
    private var previousTouchPosition: SIMD2<Float>?
    
    // Display modes matching ofxMSAFluid
    enum DisplayMode: String, CaseIterable {
        case density = "Color"      // kDrawColor
        case velocity = "Motion"    // kDrawMotion  
        case speed = "Speed"        // kDrawSpeed
        case vorticity = "Vorticity"
    }
    @Published var displayMode: DisplayMode = .density
    
    init() {
        setupMetal()
        setupPipelines()
        setupTextures()
    }
    
    private func setupMetal() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported")
        }
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        
        // Load shaders
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Failed to load Metal library")
        }
        self.library = library
    }
    
    private func setupPipelines() {
        do {
            // Create compute pipeline states
            addForcesPipeline = try createComputePipeline(functionName: "addForces")
            diffusePipeline = try createComputePipeline(functionName: "diffuse")
            advectPipeline = try createComputePipeline(functionName: "advect")
            divergencePipeline = try createComputePipeline(functionName: "divergence")
            pressureSolvePipeline = try createComputePipeline(functionName: "pressureSolve")
            subtractGradientPipeline = try createComputePipeline(functionName: "subtractGradient")
            vorticityPipeline = try createComputePipeline(functionName: "calculateVorticity")
            applyVorticityPipeline = try createComputePipeline(functionName: "applyVorticity")
            fadeDensityPipeline = try createComputePipeline(functionName: "fadeDensity")
            clearTexturePipeline = try createComputePipeline(functionName: "clearTexture")
            renderDensityPipeline = try createComputePipeline(functionName: "renderDensity")
            renderVelocityPipeline = try createComputePipeline(functionName: "renderVelocity")
            renderSpeedPipeline = try createComputePipeline(functionName: "renderSpeed")
        } catch {
            fatalError("Failed to create compute pipelines: \(error)")
        }
    }
    
    private func createComputePipeline(functionName: String) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: functionName) else {
            throw NSError(domain: "FluidSolver", code: 1, userInfo: [NSLocalizedDescriptionKey: "Function \(functionName) not found"])
        }
        return try device.makeComputePipelineState(function: function)
    }
    
    private func setupTextures() {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float,
            width: gridWidth,
            height: gridHeight,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        
        velocityTexture = device.makeTexture(descriptor: descriptor)!
        velocityTempTexture = device.makeTexture(descriptor: descriptor)!
        densityTexture = device.makeTexture(descriptor: descriptor)!
        densityTempTexture = device.makeTexture(descriptor: descriptor)!
        pressureTexture = device.makeTexture(descriptor: descriptor)!
        pressureTempTexture = device.makeTexture(descriptor: descriptor)!
        divergenceTexture = device.makeTexture(descriptor: descriptor)!
        vorticityTexture = device.makeTexture(descriptor: descriptor)!
        displayTexture = device.makeTexture(descriptor: descriptor)!
        
        clearTextures()
    }
    
    private func clearTextures() {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        let textures = [velocityTexture, velocityTempTexture, densityTexture, densityTempTexture,
                       pressureTexture, pressureTempTexture, divergenceTexture, vorticityTexture]
        
        for texture in textures {
            encoder.setComputePipelineState(clearTexturePipeline)
            encoder.setTexture(texture, index: 0)
            
            let threadsPerGrid = MTLSize(width: gridWidth, height: gridHeight, depth: 1)
            let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        }
        
        encoder.endEncoding()
        commandBuffer.commit()
    }
    
    func reset() {
        clearTextures()
        previousTouchPosition = nil
    }
    
    func applyPreset(_ preset: FluidPreset) {
        viscosity = preset.viscosity
        diffusion = preset.diffusion
        deltaTime = preset.deltaTime
        vorticityStrength = preset.vorticityStrength
        fadeSpeed = preset.fadeSpeed
        solverIterations = preset.solverIterations
        forceMultiplier = preset.forceMultiplier
        colorMultiplier = preset.colorMultiplier
        brightness = preset.brightness
        
        // If grid resolution changed, recreate textures
        if preset.gridResolution != gridWidth {
            gridWidth = preset.gridResolution
            gridHeight = preset.gridResolution
            setupTextures()
        }
    }
    
    func addForce(at position: CGPoint, windowSize: CGSize) {
        let normalizedPos = SIMD2<Float>(
            Float(position.x / windowSize.width) * Float(gridWidth),
            Float(position.y / windowSize.height) * Float(gridHeight)
        )
        
        var force = SIMD2<Float>(0, 0)
        if let previousPos = previousTouchPosition {
            force = (normalizedPos - previousPos) * forceMultiplier
            lastInputVelocity = force  // Store for particle system
        }
        
        // Generate color cycling through HSB spectrum like original
        // In ofxMSAFluid: (getElapsedFrames() % 360) / 360.0f
        let frameNumber = Int(Date().timeIntervalSinceReferenceDate * 60) // Approximate 60fps
        let hue = Float((frameNumber % 360)) / 360.0
        let color = hsbToRgb(h: hue, s: 1.0, b: 1.0)
        
        applyForce(position: normalizedPos, force: force, color: color)
        previousTouchPosition = normalizedPos
    }
    
    func endInteraction() {
        previousTouchPosition = nil
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
        
        return SIMD4(rgb + SIMD3(repeating: m), 1.0) * colorMultiplier
    }
    
    private func applyForce(position: SIMD2<Float>, force: SIMD2<Float>, color: SIMD4<Float>) {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(addForcesPipeline)
        encoder.setTexture(velocityTexture, index: 0)
        encoder.setTexture(densityTexture, index: 1)
        
        var pos = position
        var f = force
        var col = color
        var radius = forceRadius
        var dt = deltaTime
        
        encoder.setBytes(&pos, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
        encoder.setBytes(&f, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
        encoder.setBytes(&col, length: MemoryLayout<SIMD4<Float>>.size, index: 2)
        encoder.setBytes(&radius, length: MemoryLayout<Float>.size, index: 3)
        encoder.setBytes(&dt, length: MemoryLayout<Float>.size, index: 4)
        
        let threadsPerGrid = MTLSize(width: gridWidth, height: gridHeight, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        encoder.endEncoding()
        commandBuffer.commit()
    }
    
    func update() {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        let threadsPerGrid = MTLSize(width: gridWidth, height: gridHeight, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        
        // Velocity step
        // 1. Apply vorticity confinement
        if vorticityStrength > 0 {
            // Calculate vorticity
            encoder.setComputePipelineState(vorticityPipeline)
            encoder.setTexture(velocityTexture, index: 0)
            encoder.setTexture(vorticityTexture, index: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            
            // Apply vorticity force
            encoder.setComputePipelineState(applyVorticityPipeline)
            encoder.setTexture(vorticityTexture, index: 0)
            encoder.setTexture(velocityTexture, index: 1)
            var strength = vorticityStrength
            var dt = deltaTime
            encoder.setBytes(&strength, length: MemoryLayout<Float>.size, index: 0)
            encoder.setBytes(&dt, length: MemoryLayout<Float>.size, index: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        }
        
        // 2. Diffuse velocity
        if viscosity > 0 {
            for _ in 0..<solverIterations {
                encoder.setComputePipelineState(diffusePipeline)
                encoder.setTexture(velocityTexture, index: 0)
                encoder.setTexture(velocityTempTexture, index: 1)
                var visc = viscosity
                var dt = deltaTime
                encoder.setBytes(&visc, length: MemoryLayout<Float>.size, index: 0)
                encoder.setBytes(&dt, length: MemoryLayout<Float>.size, index: 1)
                encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                
                swap(&velocityTexture, &velocityTempTexture)
            }
        }
        
        // 3. Project (make incompressible)
        // Calculate divergence
        encoder.setComputePipelineState(divergencePipeline)
        encoder.setTexture(velocityTexture, index: 0)
        encoder.setTexture(divergenceTexture, index: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        // Clear pressure
        encoder.setComputePipelineState(clearTexturePipeline)
        encoder.setTexture(pressureTexture, index: 0)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        // Solve for pressure
        for _ in 0..<solverIterations {
            encoder.setComputePipelineState(pressureSolvePipeline)
            encoder.setTexture(divergenceTexture, index: 0)
            encoder.setTexture(pressureTexture, index: 1)
            encoder.setTexture(pressureTempTexture, index: 2)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            
            swap(&pressureTexture, &pressureTempTexture)
        }
        
        // Subtract pressure gradient
        encoder.setComputePipelineState(subtractGradientPipeline)
        encoder.setTexture(pressureTexture, index: 0)
        encoder.setTexture(velocityTexture, index: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        // 4. Advect velocity
        encoder.setComputePipelineState(advectPipeline)
        encoder.setTexture(velocityTexture, index: 0)
        encoder.setTexture(velocityTexture, index: 1)
        encoder.setTexture(velocityTempTexture, index: 2)
        var dt = deltaTime
        encoder.setBytes(&dt, length: MemoryLayout<Float>.size, index: 0)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        swap(&velocityTexture, &velocityTempTexture)
        
        // Density step
        // 1. Diffuse density (if needed)
        if diffusion > 0 {
            for _ in 0..<solverIterations {
                encoder.setComputePipelineState(diffusePipeline)
                encoder.setTexture(densityTexture, index: 0)
                encoder.setTexture(densityTempTexture, index: 1)
                var diff = diffusion
                encoder.setBytes(&diff, length: MemoryLayout<Float>.size, index: 0)
                encoder.setBytes(&dt, length: MemoryLayout<Float>.size, index: 1)
                encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                
                swap(&densityTexture, &densityTempTexture)
            }
        }
        
        // 2. Advect density
        encoder.setComputePipelineState(advectPipeline)
        encoder.setTexture(velocityTexture, index: 0)
        encoder.setTexture(densityTexture, index: 1)
        encoder.setTexture(densityTempTexture, index: 2)
        encoder.setBytes(&dt, length: MemoryLayout<Float>.size, index: 0)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        swap(&densityTexture, &densityTempTexture)
        
        // 3. Fade density
        encoder.setComputePipelineState(fadeDensityPipeline)
        encoder.setTexture(densityTexture, index: 0)
        var fade = fadeSpeed
        encoder.setBytes(&fade, length: MemoryLayout<Float>.size, index: 0)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        // Render to display texture
        switch displayMode {
        case .density:  // Color mode
            encoder.setComputePipelineState(renderDensityPipeline)
            encoder.setTexture(densityTexture, index: 0)
            encoder.setTexture(displayTexture, index: 1)
            var bright = brightness
            encoder.setBytes(&bright, length: MemoryLayout<Float>.size, index: 0)
        case .velocity:  // Motion mode
            encoder.setComputePipelineState(renderVelocityPipeline)
            encoder.setTexture(velocityTexture, index: 0)
            encoder.setTexture(displayTexture, index: 1)
            var scale: Float = 1.0
            encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 0)
        case .speed:  // Speed mode
            encoder.setComputePipelineState(renderSpeedPipeline)
            encoder.setTexture(velocityTexture, index: 0)
            encoder.setTexture(displayTexture, index: 1)
            var scale: Float = 25.0
            encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 0)
        case .vorticity:  // Vorticity visualization
            encoder.setComputePipelineState(renderDensityPipeline)
            encoder.setTexture(vorticityTexture, index: 0)
            encoder.setTexture(displayTexture, index: 1)
            var bright: Float = 10.0
            encoder.setBytes(&bright, length: MemoryLayout<Float>.size, index: 0)
        }
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        encoder.endEncoding()
        commandBuffer.commit()
    }
    
    var texture: MTLTexture? {
        return displayTexture
    }
    
    // Store last known velocity for particle system
    private var lastInputVelocity = SIMD2<Float>(0, 0)
    private var velocityField: [SIMD2<Float>] = []
    
    // Get velocity at normalized position for particle system
    func getVelocityAt(normalizedPos: SIMD2<Float>) -> SIMD2<Float> {
        // Clamp position to valid range
        let x = Int(max(0, min(Float(gridWidth - 1), normalizedPos.x * Float(gridWidth))))
        let y = Int(max(0, min(Float(gridHeight - 1), normalizedPos.y * Float(gridHeight))))
        
        // If we have cached velocity field data, use it
        if !velocityField.isEmpty {
            let index = y * gridWidth + x
            if index < velocityField.count {
                return velocityField[index]
            }
        }
        
        // Otherwise return last input velocity as fallback
        return lastInputVelocity * 0.01
    }
    
    // Update cached velocity field (called periodically)
    private func updateVelocityField() {
        // This would ideally read from the GPU texture
        // For now, we'll approximate based on last input
        // The real implementation would use a compute shader to copy data
    }
}