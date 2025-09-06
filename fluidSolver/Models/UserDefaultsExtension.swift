//
//  UserDefaultsExtension.swift
//  fluidSolver
//
//  Created by Aidan Cornelius-Bell on 6/9/2025.
//

import Foundation

/// Extensions to UserDefaults for persistent storage of fluid simulation settings.
///
/// Provides structured access to simulation parameters with crash recovery
/// mechanisms to prevent unstable settings from persisting after crashes.
extension UserDefaults {
    // MARK: - Keys
    
    /// Structured keys for all persistent settings.
    ///
    /// Organised into categories for app state, simulation parameters,
    /// display settings, and particle system configuration.
    enum FluidKeys: String {
        // MARK: App state
        /// Tracks whether app crashed on last run
        case didCrash = "app.didCrash"
        /// Last known stable resolution before crash
        case lastSuccessfulResolution = "app.lastSuccessfulResolution"
        
        // MARK: Simulation parameters
        /// Fluid viscosity (thickness)
        case viscosity = "fluid.viscosity"
        /// Colour diffusion rate
        case diffusion = "fluid.diffusion"
        /// Simulation time step
        case deltaTime = "fluid.deltaTime"
        /// Vorticity confinement strength
        case vorticityStrength = "fluid.vorticityStrength"
        /// Density fade rate
        case fadeSpeed = "fluid.fadeSpeed"
        /// Pressure solver iterations
        case solverIterations = "fluid.solverIterations"
        /// Grid resolution in pixels
        case gridResolution = "fluid.gridResolution"
        /// Resolution scaling factor
        case resolutionScale = "fluid.resolutionScale"
        
        // MARK: Display settings
        /// Force input multiplier
        case forceMultiplier = "fluid.forceMultiplier"
        /// Colour intensity multiplier
        case colorMultiplier = "fluid.colorMultiplier"
        /// Display brightness adjustment
        case brightness = "fluid.brightness"
        /// Force application radius
        case forceRadius = "fluid.forceRadius"
        /// Current visualisation mode
        case displayMode = "fluid.displayMode"
        /// Active colour palette
        case colorPalette = "fluid.colorPalette"
        
        // MARK: Particle settings
        /// Master particle enable
        case particlesEnabled = "particles.enabled"
        /// Particle momentum preservation
        case momentum = "particles.momentum"
        /// Fluid coupling strength
        case fluidForce = "particles.fluidForce"
        /// Particle fade rate
        case fadeSpeedParticles = "particles.fadeSpeed"
        /// Particles per spawn
        case spawnCount = "particles.spawnCount"
        /// Spawn position radius
        case spawnRadius = "particles.spawnRadius"
        /// Fluid influence decay rate
        case fluidDecayRate = "particles.fluidDecayRate"
        /// Link coupling to viscosity
        case linkToViscosity = "particles.linkToViscosity"
        /// Visual particle size
        case particleSize = "particles.particleSize"
    }
    
    // MARK: - Crash recovery
    
    /// Marks app as launched for crash detection.
    ///
    /// Called at app startup to set crash flag. If app doesn't
    /// terminate cleanly, this flag indicates a crash occurred.
    static func markAppLaunched() {
        UserDefaults.standard.set(true, forKey: FluidKeys.didCrash.rawValue)
        UserDefaults.standard.synchronize()
    }
    
    /// Marks app as terminated cleanly.
    ///
    /// Called when app enters background or terminates normally
    /// to clear the crash flag.
    static func markAppTerminatedCleanly() {
        UserDefaults.standard.set(false, forKey: FluidKeys.didCrash.rawValue)
        UserDefaults.standard.synchronize()
    }
    
    /// Checks if app crashed on previous run.
    ///
    /// - Returns: True if app didn't terminate cleanly last time
    static func didCrashLastTime() -> Bool {
        return UserDefaults.standard.bool(forKey: FluidKeys.didCrash.rawValue)
    }
    
    /// Saves a resolution that ran successfully.
    ///
    /// Used to track stable resolution settings for crash recovery.
    ///
    /// - Parameter resolution: Grid resolution that ran without issues
    static func saveSuccessfulResolution(_ resolution: Int) {
        UserDefaults.standard.set(resolution, forKey: FluidKeys.lastSuccessfulResolution.rawValue)
    }
    
    /// Gets a safe resolution considering crash history.
    ///
    /// Returns default resolution if app crashed last time,
    /// otherwise returns saved resolution preference.
    ///
    /// - Parameter defaultResolution: Fallback resolution if crashed
    /// - Returns: Safe grid resolution to use
    static func getSafeResolution(default defaultResolution: Int = 512) -> Int {
        if didCrashLastTime() {
            // Reset to safe resolution after crash
            print("App crashed last time, resetting to safe resolution")
            return defaultResolution
        }
        
        let saved = UserDefaults.standard.integer(forKey: FluidKeys.gridResolution.rawValue)
        return saved > 0 ? saved : defaultResolution
    }
    
    // MARK: - Reset
    
    /// Resets all fluid simulation settings to defaults.
    ///
    /// Removes all stored preferences, forcing the app to use
    /// default values on next launch.
    static func resetAllSettings() {
        for key in FluidKeys.allCases {
            UserDefaults.standard.removeObject(forKey: key.rawValue)
        }
        UserDefaults.standard.synchronize()
    }
}

/// Conformance to CaseIterable for easy iteration over all keys.
extension UserDefaults.FluidKeys: CaseIterable {}
