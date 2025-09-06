//
//  ColorPalette.swift
//  fluidSolver
//
//  Created by Aidan Cornelius-Bell on 6/9/2025.
//

import Foundation
import simd

/// Colour palette options for fluid rendering.
///
/// Provides various colour schemes that cycle over time to create
/// dynamic visual effects in the fluid simulation.
enum ColorPalette: String, CaseIterable {
    /// Classic HSB rainbow cycling
    case rainbow = "Rainbow"
    /// Blue to cyan ocean waves
    case ocean = "Ocean"
    /// Red to orange to yellow fire tones
    case fire = "Fire"
    /// Purple to pink to yellow plasma effect
    case plasma = "Plasma"
    /// Bright saturated neon colours
    case neon = "Neon"
    /// Orange to pink to purple sunset gradient
    case sunset = "Sunset"
    /// Vaporwave aesthetic with pink, purple and cyan
    case vapor = "Vapor"
    /// Monochrome gradient for subtle effects
    case greyscale = "Greyscale"
    
    /// Generates a colour based on the palette and time.
    ///
    /// Creates animated colour transitions that cycle smoothly over time.
    /// The time parameter is typically based on frame count or real time.
    ///
    /// - Parameter time: Time value in 0-1 range (wraps automatically)
    /// - Returns: RGBA colour vector
    func getColor(for time: Float) -> SIMD4<Float> {
        switch self {
        case .rainbow:
            // Original HSB cycling
            return hsbToRgb(h: time, s: 1.0, b: 1.0)
            
        case .ocean:
            // Blue to cyan to white
            let phase = time * 2 * .pi
            let r = 0.2 + 0.3 * sin(phase)
            let g = 0.5 + 0.4 * sin(phase + .pi/3)
            let b = 0.8 + 0.2 * sin(phase)
            return SIMD4<Float>(r, g, b, 1.0)
            
        case .fire:
            // Red to orange to yellow
            let r = 1.0
            let g = min(1.0, time * 2)
            let b = max(0, (time - 0.5) * 2)
            return SIMD4<Float>(Float(r), g, b, 1.0)
            
        case .plasma:
            // Purple to pink to yellow
            let phase = time * 2 * .pi
            let r = 0.5 + 0.5 * sin(phase)
            let g = 0.3 + 0.5 * sin(phase + 2 * .pi/3)
            let b = 0.8 + 0.2 * sin(phase + 4 * .pi/3)
            return SIMD4<Float>(r, g, b, 1.0)
            
        case .neon:
            // Bright saturated colors
            let hue = fmod(time * 1.5, 1.0)
            return hsbToRgb(h: hue, s: 1.0, b: 1.0)
            
        case .sunset:
            // Orange to pink to purple
            let phase = time * 2 * .pi
            let r = 0.9 + 0.1 * sin(phase)
            let g = 0.3 + 0.4 * sin(phase + .pi/2)
            let b = 0.4 + 0.6 * sin(phase + .pi)
            return SIMD4<Float>(r, g, b, 1.0)
            
        case .vapor:
            // Vaporwave aesthetic - pink/purple/cyan
            let phase = time * 2 * .pi
            let r = 0.7 + 0.3 * sin(phase)
            let g = 0.3 + 0.3 * sin(phase + .pi/2)
            let b = 0.9 + 0.1 * sin(phase + .pi)
            return SIMD4<Float>(r, g, b, 1.0)
            
        case .greyscale:
            // Greyscale gradient
            let brightness = 0.5 + 0.5 * sin(time * 2 * .pi)
            return SIMD4<Float>(brightness, brightness, brightness, 1.0)
        }
    }
    
    /// Converts HSB colour space to RGBA.
    ///
    /// Used internally for rainbow and neon palettes that work
    /// naturally in HSB space.
    ///
    /// - Parameters:
    ///   - h: Hue (0-1, red to red through spectrum)
    ///   - s: Saturation (0-1, grey to full colour)
    ///   - b: Brightness (0-1, black to full brightness)
    /// - Returns: RGBA colour vector with alpha of 1.0
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
