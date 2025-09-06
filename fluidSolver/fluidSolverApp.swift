import SwiftUI

@main
struct fluidSolverApp: App {
    #if os(macOS)
    @StateObject private var appState = AppState()
    #endif
    
    var body: some Scene {
        #if os(macOS)
        macOSScene
        #else
        iOSScene
        #endif
    }
    
    #if os(macOS)
    private var macOSScene: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .frame(minWidth: 800, minHeight: 600)
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified)
        .commands {
            presetCommands
        }
    }
    
    @CommandsBuilder
    private var presetCommands: some Commands {
        CommandGroup(after: .appInfo) {
            Divider()
            presetMenu
            Divider()
            resetButton
        }
    }
    
    private var presetMenu: some View {
        Menu("Presets") {
            ForEach(Array(FluidPreset.presets.prefix(9).enumerated()), id: \.element.name) { index, preset in
                Button(preset.name) {
                    appState.selectedPreset = preset.name
                }
                .keyboardShortcut(
                    KeyEquivalent(Character("\(index + 1)")),
                    modifiers: [.command]
                )
            }
            
            if FluidPreset.presets.count > 9 {
                Divider()
                ForEach(FluidPreset.presets.dropFirst(9), id: \.name) { preset in
                    Button(preset.name) {
                        appState.selectedPreset = preset.name
                    }
                }
            }
        }
    }
    
    private var resetButton: some View {
        Button("Reset Simulation") {
            appState.shouldReset = true
        }
        .keyboardShortcut("r", modifiers: [.command])
    }
    #else
    private var iOSScene: some Scene {
        WindowGroup {
            ContentView()
        }
    }
    #endif
}

#if os(macOS)
class AppState: ObservableObject {
    @Published var selectedPreset: String = "Default"
    @Published var shouldReset: Bool = false
}
#endif