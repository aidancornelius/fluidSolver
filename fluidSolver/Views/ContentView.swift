import SwiftUI
import MetalKit

struct ContentView: View {
    @StateObject private var fluidSolver = FluidSolver()
    @StateObject private var particleSystem: ParticleSystem
    @State private var showControls = true
    @State private var isInteracting = false
    #if os(macOS)
    @EnvironmentObject var appState: AppState
    #endif
    
    init() {
        let device = MTLCreateSystemDefaultDevice()!
        let particleSystem = ParticleSystem(device: device)
        _particleSystem = StateObject(wrappedValue: particleSystem)
    }
    
    var body: some View {
        mainContent
            .preferredColorScheme(.dark)
            .onAppear(perform: setupInitialState)
            #if os(macOS)
            .onChange(of: appState.selectedPreset, perform: handlePresetChange)
            .onChange(of: appState.shouldReset, perform: handleReset)
            .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
                applyDefaultPreset()
            }
            #endif
    }
    
    private var mainContent: some View {
        ZStack {
            fluidView
            controlPanelOverlay
            toggleButtonOverlay
        }
    }
    
    private var fluidView: some View {
        GeometryReader { geometry in
            MetalView(fluidSolver: fluidSolver, particleSystem: particleSystem)
                .onContinuousHover { phase in
                    handleHover(phase: phase, in: geometry.size)
                }
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            handleDrag(location: value.location, in: geometry.size)
                        }
                        .onEnded { _ in
                            fluidSolver.endInteraction()
                        }
                )
        }
        .ignoresSafeArea()
    }
    
    private var controlPanelOverlay: some View {
        HStack {
            Spacer()
            
            if showControls {
                ControlPanel(fluidSolver: fluidSolver, particleSystem: particleSystem)
                    .frame(width: 320)
                    .background(Color.black.opacity(0.85))
                    .transition(.asymmetric(
                        insertion: .move(edge: .trailing),
                        removal: .move(edge: .trailing)
                    ))
            }
        }
        .animation(.easeInOut(duration: 0.3), value: showControls)
    }
    
    private var toggleButtonOverlay: some View {
        VStack {
            HStack {
                Spacer()
                
                Button(action: toggleControls) {
                    Image(systemName: showControls ? "sidebar.trailing" : "sidebar.leading")
                        .font(.title2)
                        .foregroundColor(.white)
                        .padding(12)
                        .background(Color.black.opacity(0.7))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                }
                .padding(.trailing, showControls ? 340 : 20)  // Move button when sidebar is open
                #if os(iOS)
                .padding(.top, 60)  // More padding on iOS to avoid status bar
                #else
                .padding(.top, 20)
                #endif
                .zIndex(1000)
            }
            
            Spacer()
        }
    }
    
    private func setupInitialState() {
        if let defaultPreset = FluidPreset.presets.first {
            fluidSolver.applyPreset(defaultPreset)
        }
    }
    
    private func toggleControls() {
        withAnimation(.easeInOut(duration: 0.3)) {
            showControls.toggle()
        }
    }
    
    #if os(macOS)
    private func handlePresetChange(_ newValue: String) {
        if let preset = FluidPreset.presets.first(where: { $0.name == newValue }) {
            fluidSolver.applyPreset(preset)
            fluidSolver.reset()
        }
    }
    
    private func handleReset(_ shouldReset: Bool) {
        if shouldReset {
            fluidSolver.reset()
            particleSystem.reset()
            appState.shouldReset = false
        }
    }
    
    private func applyDefaultPreset() {
        if let defaultPreset = FluidPreset.presets.first(where: { $0.name == "Default" }) {
            fluidSolver.applyPreset(defaultPreset)
        }
    }
    #endif
    
    private func handleHover(phase: HoverPhase, in size: CGSize) {
        #if os(macOS)
        switch phase {
        case .active(let location):
            if isInteracting {
                handleInteraction(at: location, in: size)
            }
        case .ended:
            fluidSolver.endInteraction()
        }
        #endif
    }
    
    private func handleDrag(location: CGPoint, in size: CGSize) {
        isInteracting = true
        handleInteraction(at: location, in: size)
    }
    
    private func handleInteraction(at location: CGPoint, in size: CGSize) {
        fluidSolver.addForce(at: location, windowSize: size)
        
        // Add more particles continuously (matching original)
        particleSystem.addParticles(at: location, count: particleSystem.spawnCount, windowSize: size)
    }
}

struct ControlPanel: View {
    @ObservedObject var fluidSolver: FluidSolver
    @ObservedObject var particleSystem: ParticleSystem
    @State private var selectedSection = 0
    @State private var selectedPreset: String = "Default"
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            Text("Fluid Controls")
                .font(.headline)
                .foregroundColor(.white)
                .padding()
                .frame(maxWidth: .infinity)
                .background(LinearGradient(
                    colors: [Color(white: 0.2), Color(white: 0.15)],
                    startPoint: .top,
                    endPoint: .bottom
                ))
            
            // Section picker (moved to top for visibility)
            Picker("Section", selection: $selectedSection) {
                Text("Simulation").tag(0)
                Text("Visual").tag(1)
                Text("Particles").tag(2)
            }
            #if os(iOS)
            .pickerStyle(SegmentedPickerStyle())
            .padding(.horizontal)
            .padding(.vertical, 10)
            .background(Color(white: 0.1))
            #else
            .pickerStyle(SegmentedPickerStyle())
            .padding()
            #endif
            
            // Preset selector
            VStack(alignment: .leading) {
                Text("Preset")
                    .font(.caption)
                    .foregroundColor(.gray)
                
                Picker("Preset", selection: $selectedPreset) {
                    ForEach(FluidPreset.presets, id: \.name) { preset in
                        Text(preset.name).tag(preset.name)
                    }
                }
                .pickerStyle(MenuPickerStyle())
                .frame(maxWidth: .infinity)
                .onChange(of: selectedPreset) { newValue in
                    if let preset = FluidPreset.presets.first(where: { $0.name == newValue }) {
                        fluidSolver.applyPreset(preset)
                        fluidSolver.reset()
                    }
                }
            }
            .padding(.horizontal)
            .padding(.bottom)
            
            ScrollView {
                VStack(alignment: .leading, spacing: 15) {
                    switch selectedSection {
                    case 0:
                        simulationControls
                    case 1:
                        visualControls
                    case 2:
                        particleControls
                    default:
                        EmptyView()
                    }
                }
                .padding()
            }
            
            // Action buttons
            VStack(spacing: 10) {
                Button(action: {
                    fluidSolver.reset()
                    particleSystem.reset()
                }) {
                    Label("Reset Simulation", systemImage: "arrow.clockwise")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color(white: 0.25))
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
                .keyboardShortcut("r", modifiers: [.command])
                
                #if os(macOS)
                Text("Tip: ⌘R to reset, ⌘1-9 for presets")
                    .font(.caption2)
                    .foregroundColor(.gray)
                    .frame(maxWidth: .infinity)
                #endif
            }
            .padding()
        }
    }
    
    var simulationControls: some View {
        Group {
            SliderControl(
                title: "Viscosity",
                value: $fluidSolver.viscosity,
                range: 0...0.01,  // Original: 0.0 to 0.01
                format: "%.6f"
            )
            
            SliderControl(
                title: "Color Diffusion",
                value: $fluidSolver.diffusion,
                range: 0...0.0003,  // Original: 0.0 to 0.0003
                format: "%.6f"
            )
            
            SliderControl(
                title: "Delta Time",
                value: $fluidSolver.deltaTime,
                range: 0.1...5.0,  // Original: 0.1 to 5
                format: "%.2f"
            )
            
            SliderControl(
                title: "Vorticity",
                value: $fluidSolver.vorticityStrength,
                range: 0...50,  // Keeping this range for vorticity confinement
                format: "%.1f"
            )
            
            SliderControl(
                title: "Fade Speed",
                value: $fluidSolver.fadeSpeed,
                range: 0...0.1,  // Original: 0.0 to 0.1
                format: "%.4f"
            )
            
            SliderControl(
                title: "Solver Iterations",
                value: Binding(
                    get: { Double(fluidSolver.solverIterations) },
                    set: { fluidSolver.solverIterations = Int($0) }
                ),
                range: 1...50,  // Original: 1 to 50
                format: "%.0f"
            )
        }
    }
    
    var visualControls: some View {
        Group {
            SliderControl(
                title: "Velocity Multiplier",
                value: $fluidSolver.forceMultiplier,
                range: 0...100,  // Original: velocityMult 0 to 100
                format: "%.0f"
            )
            
            SliderControl(
                title: "Color Multiplier",
                value: $fluidSolver.colorMultiplier,
                range: 0...100,  // Original: colorMult 0 to 100
                format: "%.0f"
            )
            
            SliderControl(
                title: "Brightness",
                value: $fluidSolver.brightness,
                range: 0...2.0,  // Original: brightness 0.0 to 2
                format: "%.3f"
            )
            
            SliderControl(
                title: "Force Radius",
                value: $fluidSolver.forceRadius,
                range: 0.5...50,  // Much wider range with smaller values
                format: "%.1f"
            )
            
            VStack(alignment: .leading, spacing: 10) {
                Text("Draw Mode")
                    .font(.caption)
                    .foregroundColor(.gray)
                
                Picker("Draw Mode", selection: $fluidSolver.displayMode) {
                    Text("Color").tag(FluidSolver.DisplayMode.density)
                    Text("Motion").tag(FluidSolver.DisplayMode.velocity)
                    Text("Speed").tag(FluidSolver.DisplayMode.speed)
                    Text("Vorticity").tag(FluidSolver.DisplayMode.vorticity)
                }
                .pickerStyle(SegmentedPickerStyle())
                .frame(maxWidth: .infinity)
                
                // Draw mode specific info
                Group {
                    switch fluidSolver.displayMode {
                    case .density:
                        Text("Shows fluid density/color")
                    case .velocity:
                        Text("Shows velocity field as RGB")
                    case .speed:
                        Text("Shows flow speed intensity")
                    case .vorticity:
                        Text("Shows rotational motion")
                    }
                }
                .font(.caption2)
                .foregroundColor(.gray)
                .italic()
            }
        }
    }
    
    var particleControls: some View {
        Group {
            Toggle("Enable Particles", isOn: $particleSystem.isEnabled)
                .toggleStyle(SwitchToggleStyle())
            
            HStack {
                Text("Particle Count:")
                    .font(.caption)
                    .foregroundColor(.gray)
                Spacer()
                Text("\(particleSystem.particleCount)")
                    .font(.caption.monospacedDigit())
                    .foregroundColor(.white)
            }
            
            SliderControl(
                title: "Spawn Rate",
                value: Binding(
                    get: { Double(particleSystem.spawnCount) },
                    set: { particleSystem.spawnCount = Int($0) }
                ),
                range: 1...50,
                format: "%.0f"
            )
            
            SliderControl(
                title: "Spawn Radius",
                value: $particleSystem.spawnRadius,
                range: 0...30,
                format: "%.1f"
            )
            
            SliderControl(
                title: "Fluid Force",
                value: $particleSystem.fluidForce,
                range: 0...1.0,
                format: "%.2f"
            )
            
            SliderControl(
                title: "Momentum",
                value: $particleSystem.momentum,
                range: 0...1.0,
                format: "%.2f"
            )
            
            SliderControl(
                title: "Fluid Decay Rate",
                value: $particleSystem.fluidDecayRate,
                range: 0.5...10.0,
                format: "%.1f"
            )
        }
    }
}

struct SliderControl: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let format: String
    
    init(title: String, value: Binding<Float>, range: ClosedRange<Double>, format: String = "%.2f") {
        self.title = title
        self._value = Binding(
            get: { Double(value.wrappedValue) },
            set: { value.wrappedValue = Float($0) }
        )
        self.range = range
        self.format = format
    }
    
    init(title: String, value: Binding<Double>, range: ClosedRange<Double>, format: String = "%.2f") {
        self.title = title
        self._value = value
        self.range = range
        self.format = format
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.gray)
                Spacer()
                Text(String(format: format, value))
                    .font(.caption.monospacedDigit())
                    .foregroundColor(.white)
            }
            
            Slider(value: $value, in: range)
                .accentColor(.blue)
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}