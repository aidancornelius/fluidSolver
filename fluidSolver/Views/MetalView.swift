import SwiftUI
import MetalKit

#if os(macOS)
import AppKit
typealias ViewRepresentable = NSViewRepresentable
#else
import UIKit
typealias ViewRepresentable = UIViewRepresentable
#endif

struct MetalView: ViewRepresentable {
    @ObservedObject var fluidSolver: FluidSolver
    @ObservedObject var particleSystem: ParticleSystem
    
    #if os(macOS)
    func makeNSView(context: Context) -> MTKView {
        let mtkView = MTKView()
        mtkView.delegate = context.coordinator
        mtkView.preferredFramesPerSecond = 60
        mtkView.enableSetNeedsDisplay = false
        mtkView.device = MTLCreateSystemDefaultDevice()
        mtkView.framebufferOnly = false
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        mtkView.isPaused = false
        
        context.coordinator.mtkView = mtkView
        
        return mtkView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        context.coordinator.fluidSolver = fluidSolver
        context.coordinator.particleSystem = particleSystem
    }
    #else
    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView()
        mtkView.delegate = context.coordinator
        mtkView.preferredFramesPerSecond = 60
        mtkView.enableSetNeedsDisplay = false
        mtkView.device = MTLCreateSystemDefaultDevice()
        mtkView.framebufferOnly = false
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        mtkView.isPaused = false
        
        // Enable multi-touch on iOS
        mtkView.isMultipleTouchEnabled = true
        
        context.coordinator.mtkView = mtkView
        
        return mtkView
    }
    
    func updateUIView(_ uiView: MTKView, context: Context) {
        context.coordinator.fluidSolver = fluidSolver
        context.coordinator.particleSystem = particleSystem
    }
    #endif
    
    func makeCoordinator() -> Coordinator {
        Coordinator(fluidSolver: fluidSolver, particleSystem: particleSystem)
    }
    
    class Coordinator: NSObject, MTKViewDelegate {
        var fluidSolver: FluidSolver
        var particleSystem: ParticleSystem
        weak var mtkView: MTKView?
        
        private var commandQueue: MTLCommandQueue?
        private var renderPipelineState: MTLRenderPipelineState?
        
        init(fluidSolver: FluidSolver, particleSystem: ParticleSystem) {
            self.fluidSolver = fluidSolver
            self.particleSystem = particleSystem
            super.init()
            setupMetal()
        }
        
        private func setupMetal() {
            guard let device = MTLCreateSystemDefaultDevice() else { return }
            commandQueue = device.makeCommandQueue()
            
            // Create inline shader functions for texture rendering
            let shaderSource = """
            #include <metal_stdlib>
            using namespace metal;
            
            struct VertexOut {
                float4 position [[position]];
                float2 texCoord;
            };
            
            vertex VertexOut textureVertex(uint vertexID [[vertex_id]]) {
                VertexOut out;
                
                // Generate a full-screen triangle
                float2 positions[3] = {
                    float2(-1, -1),
                    float2(3, -1),
                    float2(-1, 3)
                };
                
                float2 texCoords[3] = {
                    float2(0, 1),
                    float2(2, 1),
                    float2(0, -1)
                };
                
                out.position = float4(positions[vertexID], 0, 1);
                out.texCoord = texCoords[vertexID];
                
                return out;
            }
            
            fragment float4 textureFragment(VertexOut in [[stage_in]],
                                           texture2d<float> texture [[texture(0)]]) {
                constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
                return texture.sample(textureSampler, in.texCoord);
            }
            """
            
            do {
                let library = try device.makeLibrary(source: shaderSource, options: nil)
                let vertexFunction = library.makeFunction(name: "textureVertex")
                let fragmentFunction = library.makeFunction(name: "textureFragment")
                
                let pipelineDescriptor = MTLRenderPipelineDescriptor()
                pipelineDescriptor.vertexFunction = vertexFunction
                pipelineDescriptor.fragmentFunction = fragmentFunction
                pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
                
                renderPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
            } catch {
                print("Failed to create render pipeline: \(error)")
            }
        }
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            // Handle size changes if needed
        }
        
        func draw(in view: MTKView) {
            // Update fluid simulation
            fluidSolver.update()
            
            // Update particle system
            let windowSize = CGSize(width: view.bounds.width, height: view.bounds.height)
            particleSystem.update(fluidSolver: fluidSolver, windowSize: windowSize)
            
            // Render
            guard let drawable = view.currentDrawable,
                  let commandBuffer = commandQueue?.makeCommandBuffer(),
                  let renderPassDescriptor = view.currentRenderPassDescriptor,
                  let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor),
                  let pipelineState = renderPipelineState else { return }
            
            // Draw fluid texture
            if let fluidTexture = fluidSolver.texture {
                renderEncoder.setRenderPipelineState(pipelineState)
                renderEncoder.setFragmentTexture(fluidTexture, index: 0)
                renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            }
            
            // Draw particles on top
            particleSystem.render(in: renderEncoder, viewportSize: windowSize)
            
            renderEncoder.endEncoding()
            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }
}