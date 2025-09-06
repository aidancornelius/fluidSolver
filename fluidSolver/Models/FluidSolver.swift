//
//  FluidSolver.swift
//  fluidSolver
//
//  Created by Aidan Cornelius-Bell on 6/9/2025.
//

import Foundation
import Metal
import MetalKit
import simd

/// Core fluid dynamics solver using Metal compute shaders.
///
/// Implements a Navier-Stokes fluid simulation with support for velocity advection,
/// pressure projection, vorticity confinement, and density transport. The solver
/// uses GPU compute shaders for real-time performance with grid resolutions up to
/// 3072x3072 pixels.
///
/// The simulation follows the stable fluids algorithm with enhancements for visual
/// quality including vorticity confinement and multiple display modes.
class FluidSolver: ObservableObject {
    // MARK: - Metal objects
    
    /// Metal device for GPU computation
    private var device: MTLDevice!
    /// Command queue for submitting GPU work
    private var commandQueue: MTLCommandQueue!
    /// Shader library containing compute kernels
    private var library: MTLLibrary!
    
    // MARK: - Compute pipeline states
    
    /// Pipeline for adding external forces and density
    private var addForcesPipeline: MTLComputePipelineState!
    /// Pipeline for diffusion step (viscosity)
    private var diffusePipeline: MTLComputePipelineState!
    /// Pipeline for advection (transport along velocity field)
    private var advectPipeline: MTLComputePipelineState!
    /// Pipeline for calculating velocity divergence
    private var divergencePipeline: MTLComputePipelineState!
    /// Pipeline for iterative pressure solving
    private var pressureSolvePipeline: MTLComputePipelineState!
    /// Pipeline for making velocity field incompressible
    private var subtractGradientPipeline: MTLComputePipelineState!
    /// Pipeline for calculating vorticity (curl of velocity)
    private var vorticityPipeline: MTLComputePipelineState!
    /// Pipeline for applying vorticity confinement forces
    private var applyVorticityPipeline: MTLComputePipelineState!
    /// Pipeline for density dissipation over time
    private var fadeDensityPipeline: MTLComputePipelineState!
    /// Pipeline for clearing texture contents
    private var clearTexturePipeline: MTLComputePipelineState!
    /// Pipeline for rendering density field to display
    private var renderDensityPipeline: MTLComputePipelineState!
    /// Pipeline for rendering velocity field as colours
    private var renderVelocityPipeline: MTLComputePipelineState!
    /// Pipeline for rendering speed magnitude
    private var renderSpeedPipeline: MTLComputePipelineState!
    
    // MARK: - Textures
    
    /// Current velocity field (2D vector field)
    private var velocityTexture: MTLTexture!
    /// Temporary velocity buffer for double buffering
    private var velocityTempTexture: MTLTexture!
    /// Current density/colour field
    private var densityTexture: MTLTexture!
    /// Temporary density buffer for double buffering
    private var densityTempTexture: MTLTexture!
    /// Pressure field for projection step
    private var pressureTexture: MTLTexture!
    /// Temporary pressure buffer for iterative solving
    private var pressureTempTexture: MTLTexture!
    /// Velocity divergence field
    private var divergenceTexture: MTLTexture!
    /// Vorticity (curl) field
    private var vorticityTexture: MTLTexture!
    /// Final rendered output for display
    private var displayTexture: MTLTexture!
    
    // MARK: - Grid properties
    
    /// Resolution scaling factor for high-resolution modes.
    ///
    /// Allows rendering at lower internal resolution while maintaining
    /// high output resolution for performance optimisation.
    @Published var resolutionScale: Int = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.resolutionScale.rawValue) != nil ? UserDefaults.standard.integer(forKey: UserDefaults.FluidKeys.resolutionScale.rawValue) : 1 {
        didSet {
            UserDefaults.standard.set(resolutionScale, forKey: UserDefaults.FluidKeys.resolutionScale.rawValue)
            if resolutionScale != oldValue {
                reinitialize()
            }
        }
    }
    
    /// Actual width used for computation after scaling.
    var scaledWidth: Int {
        return max(32, gridWidth / resolutionScale)
    }
    
    /// Actual height used for computation after scaling.
    var scaledHeight: Int {
        return max(32, gridHeight / resolutionScale)
    }
    
    /// Grid width in pixels for the simulation.
    ///
    /// Higher values provide more detail but require more GPU resources.
    @Published var gridWidth: Int = UserDefaults.getSafeResolution() {
        didSet {
            if gridWidth != oldValue && oldValue > 0 {
                UserDefaults.standard.set(gridWidth, forKey: UserDefaults.FluidKeys.gridResolution.rawValue)
                UserDefaults.saveSuccessfulResolution(gridWidth)
                reinitialize()
            }
        }
    }
    /// Grid height in pixels for the simulation.
    ///
    /// Typically matches gridWidth for square simulation domain.
    @Published var gridHeight: Int = UserDefaults.getSafeResolution() {
        didSet {
            if gridHeight != oldValue && oldValue > 0 {
                UserDefaults.standard.set(gridHeight, forKey: UserDefaults.FluidKeys.gridResolution.rawValue)
                UserDefaults.saveSuccessfulResolution(gridHeight)
                reinitialize()
            }
        }
    }
    
    // MARK: - Simulation parameters
    
    /// Fluid viscosity controlling momentum diffusion.
    ///
    /// Higher values create thicker, slower-moving fluids.
    /// Range: 0.0 to 0.01
    @Published var viscosity: Float = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.viscosity.rawValue) != nil ? UserDefaults.standard.float(forKey: UserDefaults.FluidKeys.viscosity.rawValue) : 0.00015 {
        didSet { UserDefaults.standard.set(viscosity, forKey: UserDefaults.FluidKeys.viscosity.rawValue) }
    }
    /// Colour diffusion rate.
    ///
    /// Controls how quickly colours blend and spread.
    /// Range: 0.0 to 0.0003
    @Published var diffusion: Float = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.diffusion.rawValue) != nil ? UserDefaults.standard.float(forKey: UserDefaults.FluidKeys.diffusion.rawValue) : 0.0 {
        didSet { UserDefaults.standard.set(diffusion, forKey: UserDefaults.FluidKeys.diffusion.rawValue) }
    }
    /// Simulation time step.
    ///
    /// Larger values speed up fluid motion but may reduce stability.
    /// Range: 0.1 to 5.0
    @Published var deltaTime: Float = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.deltaTime.rawValue) != nil ? UserDefaults.standard.float(forKey: UserDefaults.FluidKeys.deltaTime.rawValue) : 0.5 {
        didSet { UserDefaults.standard.set(deltaTime, forKey: UserDefaults.FluidKeys.deltaTime.rawValue) }
    }
    /// Vorticity confinement strength.
    ///
    /// Enhances rotational motion and prevents numerical dissipation.
    /// Range: 0.0 to 50.0
    @Published var vorticityStrength: Float = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.vorticityStrength.rawValue) != nil ? UserDefaults.standard.float(forKey: UserDefaults.FluidKeys.vorticityStrength.rawValue) : 0.0 {
        didSet { UserDefaults.standard.set(vorticityStrength, forKey: UserDefaults.FluidKeys.vorticityStrength.rawValue) }
    }
    /// Rate at which density/colour fades over time.
    ///
    /// Creates trails and prevents permanent accumulation.
    /// Range: 0.0 to 0.1
    @Published var fadeSpeed: Float = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.fadeSpeed.rawValue) != nil ? UserDefaults.standard.float(forKey: UserDefaults.FluidKeys.fadeSpeed.rawValue) : 0.002 {
        didSet { UserDefaults.standard.set(fadeSpeed, forKey: UserDefaults.FluidKeys.fadeSpeed.rawValue) }
    }
    /// Number of iterations for pressure and diffusion solvers.
    ///
    /// More iterations improve accuracy but reduce performance.
    /// Range: 1 to 50
    @Published var solverIterations: Int = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.solverIterations.rawValue) != nil ? UserDefaults.standard.integer(forKey: UserDefaults.FluidKeys.solverIterations.rawValue) : 10 {
        didSet { UserDefaults.standard.set(solverIterations, forKey: UserDefaults.FluidKeys.solverIterations.rawValue) }
    }
    /// Multiplier for input forces.
    ///
    /// Controls how strongly user interaction affects the fluid.
    /// Range: 0.0 to 100.0
    @Published var forceMultiplier: Float = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.forceMultiplier.rawValue) != nil ? UserDefaults.standard.float(forKey: UserDefaults.FluidKeys.forceMultiplier.rawValue) : 24.0 {
        didSet { UserDefaults.standard.set(forceMultiplier, forKey: UserDefaults.FluidKeys.forceMultiplier.rawValue) }
    }
    /// Multiplier for input colour intensity.
    ///
    /// Controls how vibrant added colours appear.
    /// Range: 0.0 to 100.0
    @Published var colorMultiplier: Float = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.colorMultiplier.rawValue) != nil ? UserDefaults.standard.float(forKey: UserDefaults.FluidKeys.colorMultiplier.rawValue) : 100.0 {
        didSet { UserDefaults.standard.set(colorMultiplier, forKey: UserDefaults.FluidKeys.colorMultiplier.rawValue) }
    }
    /// Display brightness adjustment.
    ///
    /// Controls overall brightness of rendered fluid.
    /// Range: 0.0 to 2.0
    @Published var brightness: Float = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.brightness.rawValue) != nil ? UserDefaults.standard.float(forKey: UserDefaults.FluidKeys.brightness.rawValue) : 0.06 {
        didSet { UserDefaults.standard.set(brightness, forKey: UserDefaults.FluidKeys.brightness.rawValue) }
    }
    /// Radius of force application in grid units.
    ///
    /// Larger values create broader force application.
    /// Range: 0.5 to 50.0
    @Published var forceRadius: Float = UserDefaults.standard.object(forKey: UserDefaults.FluidKeys.forceRadius.rawValue) != nil ? UserDefaults.standard.float(forKey: UserDefaults.FluidKeys.forceRadius.rawValue) : 5.0 {
        didSet { UserDefaults.standard.set(forceRadius, forKey: UserDefaults.FluidKeys.forceRadius.rawValue) }
    }
    
    // MARK: - Interaction state
    
    /// Previous touch/mouse position for calculating velocity
    private var previousTouchPosition: SIMD2<Float>?
    
    // MARK: - Display modes
    
    /// Available visualisation modes for the fluid simulation.
    enum DisplayMode: String, CaseIterable {
        /// Shows fluid density as colour
        case density = "Color"
        /// Shows velocity field as RGB colours
        case velocity = "Motion"
        /// Shows speed magnitude as brightness
        case speed = "Speed"
        /// Shows rotational motion intensity
        case vorticity = "Vorticity"
        /// Clears background for particle-only display
        case particlesOnly = "Particles Only"
    }
    /// Current display mode for visualisation.
    @Published var displayMode: DisplayMode = DisplayMode(rawValue: UserDefaults.standard.string(forKey: UserDefaults.FluidKeys.displayMode.rawValue) ?? "Color") ?? .density {
        didSet { UserDefaults.standard.set(displayMode.rawValue, forKey: UserDefaults.FluidKeys.displayMode.rawValue) }
    }
    
    /// Active colour palette for fluid rendering.
    @Published var colorPalette: ColorPalette = ColorPalette(rawValue: UserDefaults.standard.string(forKey: UserDefaults.FluidKeys.colorPalette.rawValue) ?? "Rainbow") ?? .rainbow {
        didSet { UserDefaults.standard.set(colorPalette.rawValue, forKey: UserDefaults.FluidKeys.colorPalette.rawValue) }
    }
    
    /// Initialises the fluid solver with Metal setup.
    ///
    /// Creates Metal device, command queue, compute pipelines,
    /// and allocates textures for simulation.
    init() {
        setupMetal()
        setupPipelines()
        setupTextures()
    }
    
    /// Sets the simulation grid resolution.
    ///
    /// Updates both width and height to maintain square grid.
    /// Triggers texture reallocation and simulation reset.
    ///
    /// - Parameter resolution: Size in pixels for both dimensions
    func setResolution(_ resolution: Int) {
        gridWidth = resolution
        gridHeight = resolution
    }
    
    /// Reinitialises textures and resets simulation.
    ///
    /// Called when resolution or scale changes.
    private func reinitialize() {
        setupTextures()
        reset()
    }
    
    /// Configures Metal device and command queue.
    ///
    /// Initialises core Metal objects required for GPU computation.
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
    
    /// Creates compute pipeline states from shader functions.
    ///
    /// Compiles and configures all GPU kernels for fluid simulation steps.
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
    
    /// Creates a compute pipeline state from a shader function.
    ///
    /// - Parameter functionName: Name of the Metal shader function
    /// - Returns: Compiled compute pipeline state
    /// - Throws: Error if function not found or compilation fails
    private func createComputePipeline(functionName: String) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: functionName) else {
            throw NSError(domain: "FluidSolver", code: 1, userInfo: [NSLocalizedDescriptionKey: "Function \(functionName) not found"])
        }
        return try device.makeComputePipelineState(function: function)
    }
    
    /// Allocates GPU textures for simulation fields.
    ///
    /// Creates double-buffered textures for velocity and density,
    /// plus auxiliary textures for pressure solving and display.
    private func setupTextures() {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float,
            width: scaledWidth,
            height: scaledHeight,
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
    
    /// Clears all simulation textures to zero.
    ///
    /// Uses GPU kernel for efficient parallel clearing.
    private func clearTextures() {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        let textures = [velocityTexture, velocityTempTexture, densityTexture, densityTempTexture,
                       pressureTexture, pressureTempTexture, divergenceTexture, vorticityTexture]
        
        for texture in textures {
            encoder.setComputePipelineState(clearTexturePipeline)
            encoder.setTexture(texture, index: 0)
            
            let threadsPerGrid = MTLSize(width: scaledWidth, height: scaledHeight, depth: 1)
            let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        }
        
        encoder.endEncoding()
        commandBuffer.commit()
    }
    
    /// Resets the simulation to initial state.
    ///
    /// Clears all fields and resets interaction tracking.
    func reset() {
        clearTextures()
        previousTouchPosition = nil
    }
    
    /// Applies a fluid preset configuration.
    ///
    /// Updates simulation parameters to match preset values.
    /// Resolution is not changed to preserve user preference.
    ///
    /// - Parameter preset: Preset configuration to apply
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
        
        // Don't change resolution on preset change - user controls it separately
        // if preset.gridResolution != gridWidth {
        //     gridWidth = preset.gridResolution
        //     gridHeight = preset.gridResolution
        //     setupTextures()
        // }
    }
    
    /// Adds force and colour at the specified position.
    ///
    /// Calculates velocity from position delta and applies force
    /// with colour based on the active palette.
    ///
    /// - Parameters:
    ///   - position: Touch/mouse position in window coordinates
    ///   - windowSize: Current window dimensions for normalisation
    func addForce(at position: CGPoint, windowSize: CGSize) {
        let normalizedPos = SIMD2<Float>(
            Float(position.x / windowSize.width) * Float(scaledWidth),
            Float(position.y / windowSize.height) * Float(scaledHeight)
        )
        
        var force = SIMD2<Float>(0, 0)
        if let previousPos = previousTouchPosition {
            force = (normalizedPos - previousPos) * forceMultiplier
            lastInputVelocity = force  // Store for particle system
        }
        
        // Generate color based on selected palette
        let frameNumber = Int(Date().timeIntervalSinceReferenceDate * 60) // Approximate 60fps
        let time = Float((frameNumber % 360)) / 360.0
        let color = colorPalette.getColor(for: time) * colorMultiplier
        
        applyForce(position: normalizedPos, force: force, color: color)
        previousTouchPosition = normalizedPos
    }
    
    /// Ends user interaction and resets tracking.
    func endInteraction() {
        previousTouchPosition = nil
    }
    
    /// Applies force and colour to the simulation grid.
    ///
    /// Uses GPU kernel to add Gaussian-weighted force and colour
    /// at the specified position.
    ///
    /// - Parameters:
    ///   - position: Grid position for force application
    ///   - force: Velocity force vector to apply
    ///   - color: RGBA colour to add to density field
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
        
        let threadsPerGrid = MTLSize(width: scaledWidth, height: scaledHeight, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        encoder.endEncoding()
        commandBuffer.commit()
    }
    
    /// Performs one simulation timestep.
    ///
    /// Executes the full fluid simulation pipeline including
    /// velocity advection, pressure projection, density transport,
    /// and rendering to display texture.
    func update() {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        let threadsPerGrid = MTLSize(width: scaledWidth, height: scaledHeight, depth: 1)
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
        case .particlesOnly:  // Particles only - clear background
            encoder.setComputePipelineState(clearTexturePipeline)
            encoder.setTexture(displayTexture, index: 0)
        }
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        encoder.endEncoding()
        commandBuffer.commit()
    }
    
    /// Returns the current display texture for rendering.
    var texture: MTLTexture? {
        return displayTexture
    }
    
    // MARK: - Particle system integration
    
    /// Last input velocity for particle system reference
    private var lastInputVelocity = SIMD2<Float>(0, 0)
    /// Cached velocity field data (currently unused)
    private var velocityField: [SIMD2<Float>] = []
    
    /// Gets velocity at a normalised position.
    ///
    /// Used by particle system to query fluid velocity.
    /// Currently returns approximated value based on last input.
    ///
    /// - Parameter normalizedPos: Position in 0-1 range
    /// - Returns: Velocity vector at the position
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
    
    /// Updates cached velocity field from GPU texture.
    ///
    /// Placeholder for future implementation that would read
    /// velocity data from GPU for CPU-side queries.
    private func updateVelocityField() {
        // This would ideally read from the GPU texture
        // For now, we'll approximate based on last input
        // The real implementation would use a compute shader to copy data
    }
}
