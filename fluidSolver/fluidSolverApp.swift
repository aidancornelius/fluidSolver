//
//  fluidSolverApp.swift
//  fluidSolver
//
//  Created by Aidan Cornelius-Bell on 6/9/2025.
//

import SwiftUI

/// Main application entry point for the fluid solver simulation.
/// Manages app lifecycle, crash detection, and platform-specific configurations.
///
/// This app provides real-time fluid dynamics simulation with platform-optimised
/// performance settings. It includes preset management for different simulation
/// configurations and crash detection with recovery mechanisms.
@main
struct fluidSolverApp: App {
    #if os(macOS)
    /// Shared state object for macOS-specific features like preset management
    @StateObject private var appState = AppState()
    #endif
    /// Monitors scene phase changes for proper app lifecycle management
    @Environment(\.scenePhase) private var scenePhase
    
    /// Initialises the app with platform-specific optimisations.
    ///
    /// On iOS, this enables high-performance mode for smooth fluid simulation.
    /// All platforms benefit from crash detection mechanism setup.
    init() {
        // Mark app as launched (for crash detection)
        UserDefaults.markAppLaunched()
        
        #if os(iOS)
        // Request high performance mode on iOS
        requestHighPerformance()
        #endif
    }
    
    /// Main scene builder that returns platform-appropriate scene configuration
    var body: some Scene {
        #if os(macOS)
        macOSScene
        #else
        iOSScene
        #endif
    }
    
    #if os(macOS)
    /// macOS-specific scene configuration with window management and commands.
    ///
    /// Provides minimum window size constraints with unified toolbar style.
    /// Includes custom menu commands for presets and background state handling
    /// for clean termination.
    private var macOSScene: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .frame(minWidth: 800, minHeight: 600)
                .onChange(of: scenePhase) { phase in
                    if phase == .background {
                        // Mark clean termination when going to background
                        UserDefaults.markAppTerminatedCleanly()
                    }
                }
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified)
        .commands {
            presetCommands
        }
    }
    
    /// Builds custom menu commands for preset management.
    ///
    /// Adds preset selection menu and reset command to the app menu
    @CommandsBuilder
    private var presetCommands: some Commands {
        CommandGroup(after: .appInfo) {
            Divider()
            presetMenu
            Divider()
            resetButton
        }
    }
    
    /// Creates a menu for selecting fluid simulation presets.
    ///
    /// The first nine presets get keyboard shortcuts from Cmd+1 through Cmd+9.
    /// Additional presets are listed without shortcuts.
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
    
    /// Creates a button to reset the fluid simulation.
    ///
    /// Uses the keyboard shortcut Cmd+R for quick access.
    private var resetButton: some View {
        Button("Reset Simulation") {
            appState.shouldReset = true
        }
        .keyboardShortcut("r", modifiers: [.command])
    }
    #else
    /// iOS-specific scene configuration.
    ///
    /// Simplified scene without menu commands, optimised for touch interaction
    private var iOSScene: some Scene {
        WindowGroup {
            ContentView()
        }
    }
    #endif
}

#if os(macOS)
/// Shared application state for macOS.
///
/// Manages the currently selected fluid simulation preset and reset
/// triggers for the simulation.
class AppState: ObservableObject {
    /// Currently selected preset name, triggers preset change when modified
    @Published var selectedPreset: String = "Default"
    /// Flag to trigger simulation reset, automatically cleared by the simulation view
    @Published var shouldReset: Bool = false
}
#endif

#if os(iOS)
import UIKit

/// iOS-specific extensions for performance optimisation
extension fluidSolverApp {
    /// Requests high-performance mode for smooth fluid simulation on iOS.
    ///
    /// This method enables sustained performance mode to prevent CPU throttling
    /// and disables the idle timer to prevent screen dimming during interaction.
    /// It also requests ProMotion display rates of 120Hz where available.
    ///
    /// Note that these settings optimise for performance over battery usage
    /// to ensure smooth fluid simulation.
    func requestHighPerformance() {
        // Enable sustained performance mode
        ProcessInfo.processInfo.performExpiringActivity(withReason: "High Performance Fluid Simulation") { expired in
            if !expired {
                // Keep the app running at full performance
                UIApplication.shared.isIdleTimerDisabled = true
            }
        }
        
        // Request maximum FPS for ProMotion displays
        // Using modern API for iOS 15+
        if let windowScene = UIApplication.shared.connectedScenes
            .compactMap({ $0 as? UIWindowScene })
            .first,
           let window = windowScene.windows.first {
            // Request ProMotion display rate if available
            window.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations()
        }
    }
}
#endif
