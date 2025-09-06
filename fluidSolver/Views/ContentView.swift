//
//  ContentView.swift
//  fluidSolver
//
//  Created by Aidan Cornelius-Bell on 6/9/2025.
//

import SwiftUI
import MetalKit
#if os(macOS)
import AppKit
#else
import UIKit
#endif

/// Main content view for the fluid simulation interface.
///
/// This view manages the fluid solver and particle system, providing an interactive
/// canvas for fluid dynamics simulation with touch and mouse input. It includes
/// a collapsible control panel for adjusting simulation parameters and visual settings.
struct ContentView: View {
    /// Core fluid dynamics solver managing the simulation state
    @StateObject private var fluidSolver = FluidSolver()
    /// Particle system for visual enhancement of fluid motion
    @StateObject private var particleSystem: ParticleSystem
    /// Controls visibility of the settings panel
    @State private var showControls = true
    /// Tracks whether user is currently interacting with the fluid
    @State private var isInteracting = false
    #if os(macOS)
    /// Shared app state for preset management on macOS
    @EnvironmentObject var appState: AppState
    #endif
    
    /// Initialises the view with Metal device setup.
    ///
    /// Creates the Metal device and initialises the particle system.
    /// The fluid solver is initialised as a @StateObject property wrapper.
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
    
    /// Main content stack combining fluid view with overlays.
    ///
    /// Layers the Metal rendering view with control panel and toggle button overlays.
    private var mainContent: some View {
        ZStack {
            fluidView
            controlPanelOverlay
            toggleButtonOverlay
        }
    }
    
    /// Interactive fluid simulation view with gesture handling.
    ///
    /// Provides touch and mouse interaction for adding forces to the fluid.
    /// Gestures are normalised to window coordinates for consistent behaviour.
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
    
    /// Sliding control panel overlay for simulation settings.
    ///
    /// Animates in from the trailing edge when visible.
    /// Contains all simulation, visual, and particle controls.
    private var controlPanelOverlay: some View {
        HStack {
            Spacer()
            
            if showControls {
                ControlPanel(fluidSolver: fluidSolver, particleSystem: particleSystem, resetToDefaults: resetToDefaults)
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
    
    /// Toggle button for showing/hiding the control panel.
    ///
    /// Positioned in the top-right corner and moves when panel is visible
    /// to avoid being obscured.
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
    
    /// Sets up the initial simulation state with default preset.
    ///
    /// Called when the view appears to ensure consistent starting state.
    private func setupInitialState() {
        if let defaultPreset = FluidPreset.presets.first {
            fluidSolver.applyPreset(defaultPreset)
        }
    }
    
    /// Toggles the visibility of the control panel with animation.
    private func toggleControls() {
        withAnimation(.easeInOut(duration: 0.3)) {
            showControls.toggle()
        }
    }
    
    #if os(macOS)
    /// Handles preset selection changes from the app menu.
    ///
    /// Applies the selected preset and resets the simulation to show
    /// the new configuration immediately.
    private func handlePresetChange(_ newValue: String) {
        if let preset = FluidPreset.presets.first(where: { $0.name == newValue }) {
            fluidSolver.applyPreset(preset)
            fluidSolver.reset()
        }
    }
    
    /// Handles reset command from the app menu.
    ///
    /// Resets both fluid solver and particle system when triggered.
    private func handleReset(_ shouldReset: Bool) {
        if shouldReset {
            fluidSolver.reset()
            particleSystem.reset()
            appState.shouldReset = false
        }
    }
    
    /// Applies the default preset when app becomes active.
    ///
    /// Ensures consistent state when returning to the app.
    private func applyDefaultPreset() {
        if let defaultPreset = FluidPreset.presets.first(where: { $0.name == "Default" }) {
            fluidSolver.applyPreset(defaultPreset)
        }
    }
    #endif
    
    /// Handles mouse hover events for continuous interaction.
    ///
    /// Only processes hover when user is actively interacting (mouse down).
    /// Normalises coordinates to simulation space.
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
    
    /// Handles drag gestures for fluid interaction.
    ///
    /// Sets interaction state and processes the touch/mouse location.
    private func handleDrag(location: CGPoint, in size: CGSize) {
        isInteracting = true
        handleInteraction(at: location, in: size)
    }
    
    /// Processes user interaction at the specified location.
    ///
    /// Adds force to the fluid solver and spawns particles at the interaction point.
    /// Coordinates are normalised to window size for consistent behaviour.
    private func handleInteraction(at location: CGPoint, in size: CGSize) {
        fluidSolver.addForce(at: location, windowSize: size)
        
        // Add more particles continuously (matching original)
        particleSystem.addParticles(at: location, count: particleSystem.spawnCount, windowSize: size)
    }
    
    /// Resets all settings to their default values.
    ///
    /// Clears user defaults and restores original simulation parameters.
    /// Triggers a full reset of both fluid and particle systems.
    private func resetToDefaults() {
        // Reset all settings to defaults
        UserDefaults.resetAllSettings()
        
        // Reload default values
        fluidSolver.viscosity = 0.00015
        fluidSolver.diffusion = 0.0
        fluidSolver.deltaTime = 0.5
        fluidSolver.vorticityStrength = 0.0
        fluidSolver.fadeSpeed = 0.002
        fluidSolver.solverIterations = 10
        fluidSolver.forceMultiplier = 24.0
        fluidSolver.colorMultiplier = 100.0
        fluidSolver.brightness = 0.06
        fluidSolver.forceRadius = 5.0
        fluidSolver.displayMode = .density
        fluidSolver.setResolution(512)
        
        particleSystem.isEnabled = true
        particleSystem.momentum = 0.5
        particleSystem.fluidForce = 0.6
        particleSystem.fadeSpeed = 0.001
        particleSystem.spawnCount = 10
        particleSystem.spawnRadius = 15.0
        particleSystem.fluidDecayRate = 3.0
        particleSystem.linkToViscosity = true
        particleSystem.particleSize = 1.0
        
        // Reset the simulation
        fluidSolver.reset()
        particleSystem.reset()
    }
}

/// Control panel view for adjusting simulation parameters.
///
/// Provides a tabbed interface with sections for simulation physics,
/// visual settings, and particle system controls. Includes preset
/// management and reset functionality.
struct ControlPanel: View {
    /// Reference to the fluid solver for parameter binding
    @ObservedObject var fluidSolver: FluidSolver
    /// Reference to the particle system for parameter binding
    @ObservedObject var particleSystem: ParticleSystem
    /// Callback to reset all settings to defaults
    let resetToDefaults: () -> Void
    /// Currently selected control section (0: Simulation, 1: Visual, 2: Particles)
    @State private var selectedSection = 0
    /// Currently selected preset name
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
            .pickerStyle(SegmentedPickerStyle())
            #if os(iOS)
            .padding(.horizontal)
            .padding(.vertical, 12)
            .background(Color(white: 0.1))
            // Ensure the picker is interactive on iOS
            .zIndex(1)
            #else
            .padding()
            #endif
            .onChange(of: selectedSection) { newValue in
                // Force UI update on iOS
                print("Section changed to: \(newValue)")
            }
            
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
                .animation(.easeInOut(duration: 0.2), value: selectedSection)
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
                
                Button(action: resetToDefaults) {
                    Label("Reset to Defaults", systemImage: "arrow.uturn.backward")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color(white: 0.15))
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
                .keyboardShortcut("d", modifiers: [.command, .shift])
                
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
    
    /// Simulation physics controls section.
    ///
    /// Contains sliders for viscosity, diffusion, time step, vorticity,
    /// fade speed, solver iterations, and resolution settings.
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
            
            VStack(alignment: .leading, spacing: 10) {
                Text("Resolution")
                    .font(.caption)
                    .foregroundColor(.gray)
                
                Picker("Resolution", selection: Binding(
                    get: { fluidSolver.gridWidth },
                    set: { newValue in
                        if newValue == -1 {
                            // Native resolution - use screen size
                            #if os(macOS)
                            if let screen = NSScreen.main {
                                let scale = Int(screen.backingScaleFactor)
                                let nativeHeight = Int(screen.frame.height) * scale
                                fluidSolver.setResolution(nativeHeight)
                            }
                            #else
                            let screen = UIScreen.main
                            let scale = Int(screen.scale)
                            let nativeHeight = Int(screen.bounds.height) * scale
                            fluidSolver.setResolution(nativeHeight)
                            #endif
                        } else {
                            fluidSolver.setResolution(newValue)
                        }
                    }
                )) {
                    Text("64×64 (Fast)").tag(64)
                    Text("128×128 (Balanced)").tag(128)
                    Text("256×256 (Detailed)").tag(256)
                    Text("512×512 (High)").tag(512)
                    Divider()
                    Text("1024×1024 (Ultra)").tag(1024)
                    Text("1536×1536 (Extreme)").tag(1536)
                    Text("3072×3072 (Maximum)").tag(3072)
                    Text("Native (Retina)").tag(-1)
                }
                .pickerStyle(MenuPickerStyle())
                .frame(maxWidth: .infinity)
                
                Group {
                    if fluidSolver.gridWidth >= 3072 {
                        Text("⚠️ Maximum resolution - significant GPU usage")
                            .font(.caption2)
                            .foregroundColor(.red)
                    } else if fluidSolver.gridWidth > 512 {
                        Text("⚠️ High resolution - may impact performance")
                            .font(.caption2)
                            .foregroundColor(.orange)
                    } else {
                        Text("Higher resolution = more detail but slower")
                            .font(.caption2)
                            .foregroundColor(.gray)
                    }
                }
                .italic()
            }
            
            // Resolution scaler for high resolution modes
            if fluidSolver.gridWidth >= 1024 {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Resolution Scale")
                        .font(.caption)
                        .foregroundColor(.gray)
                    
                    #if os(iOS)
                    // Use HStack of buttons on iOS for better touch handling
                    HStack(spacing: 0) {
                        ForEach([1, 2, 3], id: \.self) { scale in
                            Button(action: {
                                fluidSolver.resolutionScale = scale
                            }) {
                                Text(scale == 1 ? "1x (Full)" : scale == 2 ? "2x (Half)" : "3x (Third)")
                                    .font(.caption)
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 8)
                                    .background(fluidSolver.resolutionScale == scale ? Color.blue : Color(white: 0.2))
                                    .foregroundColor(.white)
                            }
                        }
                    }
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color(white: 0.3), lineWidth: 1)
                    )
                    #else
                    Picker("Scale", selection: $fluidSolver.resolutionScale) {
                        Text("1x (Full)").tag(1)
                        Text("2x (Half)").tag(2)
                        Text("3x (Third)").tag(3)
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    #endif
                    
                    Text("Same as: \(fluidSolver.scaledWidth)×\(fluidSolver.scaledHeight)")
                        .font(.caption2)
                        .foregroundColor(.gray)
                        .italic()
                }
            }
        }
    }
    
    /// Visual appearance controls section.
    ///
    /// Contains sliders for velocity and colour multipliers, brightness,
    /// force radius, display mode selection, and colour palette options.
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
                    Text("Particles").tag(FluidSolver.DisplayMode.particlesOnly)
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
                    case .particlesOnly:
                        Text("Shows only particles")
                    }
                }
                .font(.caption2)
                .foregroundColor(.gray)
                .italic()
            }
            
            // Color palette selector
            VStack(alignment: .leading, spacing: 10) {
                Text("Color Palette")
                    .font(.caption)
                    .foregroundColor(.gray)
                
                Picker("Color Palette", selection: $fluidSolver.colorPalette) {
                    ForEach(ColorPalette.allCases, id: \.self) { palette in
                        Text(palette.rawValue).tag(palette)
                    }
                }
                .pickerStyle(MenuPickerStyle())
                .frame(maxWidth: .infinity)
            }
        }
    }
    
    /// Particle system controls section.
    ///
    /// Contains toggles and sliders for particle behaviour including
    /// spawn rate, size, momentum, and fluid interaction settings.
    var particleControls: some View {
        Group {
            Toggle("Enable Particles", isOn: $particleSystem.isEnabled)
                .toggleStyle(SwitchToggleStyle())
            
            Toggle("Link to Viscosity", isOn: $particleSystem.linkToViscosity)
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
                title: "Particle Size",
                value: $particleSystem.particleSize,
                range: 0.1...5.0,
                format: "%.1f"
            )
            
            if !particleSystem.linkToViscosity {
                SliderControl(
                    title: "Fluid Force",
                    value: $particleSystem.fluidForce,
                    range: 0...1.0,
                    format: "%.2f"
                )
            }
            
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

/// Reusable slider control component with label and value display.
///
/// Provides a consistent interface for numeric parameter adjustment
/// with support for both Float and Double bindings.
struct SliderControl: View {
    /// Display label for the control
    let title: String
    /// Bound value being controlled
    @Binding var value: Double
    /// Valid range for the slider
    let range: ClosedRange<Double>
    /// Format string for value display
    let format: String
    
    /// Initialises slider control with Float binding.
    ///
    /// Converts Float binding to Double for internal consistency.
    init(title: String, value: Binding<Float>, range: ClosedRange<Double>, format: String = "%.2f") {
        self.title = title
        self._value = Binding(
            get: { Double(value.wrappedValue) },
            set: { value.wrappedValue = Float($0) }
        )
        self.range = range
        self.format = format
    }
    
    /// Initialises slider control with Double binding.
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

/// Preview provider for ContentView in Xcode canvas.
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
