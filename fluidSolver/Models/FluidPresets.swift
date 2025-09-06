//
//  FluidPresets.swift
//  fluidSolver
//
//  Created by Aidan Cornelius-Bell on 6/9/2025.
//

import Foundation

struct FluidPreset {
    let name: String
    let viscosity: Float
    let diffusion: Float
    let deltaTime: Float
    let vorticityStrength: Float
    let fadeSpeed: Float
    let solverIterations: Int
    let forceMultiplier: Float
    let colorMultiplier: Float
    let brightness: Float
    let doVorticityConfinement: Bool
    let gridResolution: Int
}

extension FluidPreset {
    // Based on ofxMSAFluid example settings
    static let presets: [FluidPreset] = [
        FluidPreset(
            name: "Default",
            viscosity: 0.00001,  // From XML settings
            diffusion: 0.0,
            deltaTime: 0.59,  // From XML settings
            vorticityStrength: 0.0,
            fadeSpeed: 0.005,  // From XML settings
            solverIterations: 15,  // From XML settings
            forceMultiplier: 24.0,  // velocityMult from XML
            colorMultiplier: 100.0,  // colorMult from XML
            brightness: 0.06,  // From XML settings
            doVorticityConfinement: false,
            gridResolution: 177  // fluidCellsX from XML
        ),
        
        FluidPreset(
            name: "Viscous",
            viscosity: 0.001,
            diffusion: 0.0,
            deltaTime: 0.5,
            vorticityStrength: 0.0,
            fadeSpeed: 0.003,
            solverIterations: 20,
            forceMultiplier: 30.0,
            colorMultiplier: 80.0,
            brightness: 0.08,
            doVorticityConfinement: false,
            gridResolution: 150
        ),
        
        FluidPreset(
            name: "Turbulent",
            viscosity: 0.00001,
            diffusion: 0.0,
            deltaTime: 0.8,
            vorticityStrength: 25.0,
            fadeSpeed: 0.002,
            solverIterations: 10,
            forceMultiplier: 50.0,
            colorMultiplier: 120.0,
            brightness: 0.1,
            doVorticityConfinement: true,
            gridResolution: 200
        ),
        
        FluidPreset(
            name: "Smoke",
            viscosity: 0.00005,
            diffusion: 0.00001,
            deltaTime: 0.4,
            vorticityStrength: 10.0,
            fadeSpeed: 0.008,
            solverIterations: 15,
            forceMultiplier: 35.0,
            colorMultiplier: 60.0,
            brightness: 0.15,
            doVorticityConfinement: true,
            gridResolution: 128
        ),
        
        FluidPreset(
            name: "Water",
            viscosity: 0.000001,
            diffusion: 0.0,
            deltaTime: 1.0,
            vorticityStrength: 5.0,
            fadeSpeed: 0.001,
            solverIterations: 25,
            forceMultiplier: 40.0,
            colorMultiplier: 90.0,
            brightness: 0.07,
            doVorticityConfinement: true,
            gridResolution: 150
        ),
        
        FluidPreset(
            name: "Ink",
            viscosity: 0.0001,
            diffusion: 0.00005,
            deltaTime: 0.6,
            vorticityStrength: 15.0,
            fadeSpeed: 0.004,
            solverIterations: 18,
            forceMultiplier: 45.0,
            colorMultiplier: 150.0,
            brightness: 0.12,
            doVorticityConfinement: true,
            gridResolution: 180
        ),
        
        FluidPreset(
            name: "Fast Flow",
            viscosity: 0.000001,
            diffusion: 0.0,
            deltaTime: 2.0,
            vorticityStrength: 8.0,
            fadeSpeed: 0.01,
            solverIterations: 8,
            forceMultiplier: 60.0,
            colorMultiplier: 80.0,
            brightness: 0.05,
            doVorticityConfinement: true,
            gridResolution: 100
        ),
        
        FluidPreset(
            name: "High Detail",
            viscosity: 0.00001,
            diffusion: 0.0,
            deltaTime: 0.5,
            vorticityStrength: 12.0,
            fadeSpeed: 0.003,
            solverIterations: 30,
            forceMultiplier: 30.0,
            colorMultiplier: 100.0,
            brightness: 0.08,
            doVorticityConfinement: true,
            gridResolution: 256
        ),
        
        FluidPreset(
            name: "Paint",
            viscosity: 0.0005,
            diffusion: 0.0001,
            deltaTime: 0.3,
            vorticityStrength: 0.0,
            fadeSpeed: 0.0005,
            solverIterations: 20,
            forceMultiplier: 20.0,
            colorMultiplier: 200.0,
            brightness: 0.2,
            doVorticityConfinement: false,
            gridResolution: 150
        ),
        
        FluidPreset(
            name: "Minimal",
            viscosity: 0.0,
            diffusion: 0.0,
            deltaTime: 0.5,
            vorticityStrength: 0.0,
            fadeSpeed: 0.02,
            solverIterations: 5,
            forceMultiplier: 50.0,
            colorMultiplier: 50.0,
            brightness: 0.1,
            doVorticityConfinement: false,
            gridResolution: 80
        )
    ]
}
