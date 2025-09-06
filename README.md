# fluidSolver

A real-time GPU-accelerated fluid dynamics simulation. Universal for all of Apple's major platforms (iOS, iPadOS, macOS [Mac Catalyst]).

## What is this codebase?

This codebase is a real-time fluid dynamics simulator built for macOS using Swift and Metal. It's a (sort of[^1]) port of the [ofxMSAFluid](https://github.com/memoakten/ofxMSAFluid) library, built for Apple's graphics stack. The app creates mesmerising, interactive fluid simulations which respond to mouse or touch input, producing swirling patterns of colour and motion that feel alive and organic.

At its core, this is an implementation of Jos Stam's "stable fluids" algorithm. Which was a breakthrough technique from the 90s which made real-time fluid simulation practical for computer graphics. The simulation runs entirely on the GPU using Metal compute shaders, achieving something resembling 60fps performance even at resolutions up to 3072×3072 pixels, if you have the graphics grunt. You can interact with the fluid by moving the mouse or finger through the view, injecting coloured dye and forces that create turbulent, even beautiful, flows. Or just ADHD eye candy.

The app features a control panel which exposes the physics parameters, allowing you to experiment with different fluid behaviours, from thin, water-like flows to thick, viscous substances. There's also an optional particle system that adds thousands of tiny white tracers which follow fluid currents, creating fun visual trails that highlight the complex motion patterns.

## How it works mathematically

### The Navier-Stokes equations

The foundation of this simulation is the [Navier-Stokes equations](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations), which describe how fluids move through space. These partial differential equations capture the fundamental physics of fluid motion.

### The stable fluids algorithm

Jos Stam's broke these equations into a series of steps that can be computed efficiently. The algorithm treats the fluid as a grid of cells, each storing velocity and density (colour) values. Each frame, it performs these steps:

**Add forces**

User input adds velocity and dye at interaction points using Gaussian falloff.

**Diffusion step** 

This models viscosity through the heat equation. The implementation uses an implicit solver for numerical stability, iteratively solving:

```
(I - νΔt∇²)u_new = u_old
```

**Projection step**

To enforce incompressibility, the algorithm:

1. Calculates divergence: `div = -0.5 * (∂u_x/∂x + ∂u_y/∂y)`
2. Solves the Poisson equation: `∇²p = div` using Jacobi iteration
3. Subtracts the pressure gradient: `u = u - ∇p`

**Advection step**

Uses backward particle tracing—for each grid cell, it traces backwards along the velocity field to find where the fluid came from, then samples that location using bilinear interpolation. This unconditionally stable method allows large time steps without blow-up.

**Vorticity confinement**

Calculates the curl (vorticity) of the velocity field:

```
ω = ∂v/∂x - ∂u/∂y
```

Then applies a force perpendicular to the vorticity gradient, which reinforces rotational motion that would otherwise be dampened by numerical dissipation.

### GPU parallelisation

Every operation is implemented as a Metal compute kernel, with each thread processing one grid cell independently. The shaders use texture memory for 2D spatial access patterns and hardware-accelerated bilinear filtering. Double buffering prevents read-write conflicts during updates.

The particle system adds another layer, each particle samples the velocity field and follows it with decreasing influence over time, transitioning from fluid-coupled motion to ballistic trajectories. This hybrid approach creates visually rich behaviour where particles initially cluster in vortices then... spray? outward.

## Why it's fun

This simulation brings the beauty of fluid dynamics to life in an interactive way. There's something satisfying about dragging your finger across the screen and watching colourful smoke billow and swirl in response, I've been intrigued by it since I first saw the original library in action in the [late 2000s](https://vimeo.com/4446798).

The mathematics enable behaviour which feels organic. Simple interactions can produce complex results.The real-time nature makes it feel responsive and immediate. Unlike pre-rendered animations, you're directly manipulating the physics, feeling the weight and flow of the virtual fluid. Adjusting parameters like viscosity or vorticity strength can completely change the character of the simulation.

Anyway, it's a gateway into computational fluid dynamics. The code demonstrates advanced graphics programming techniques including compute shaders, texture sampling, parallel algorithms, while the visual feedback makes the abstract mathematics tangible. Whatever the draw, physics, programming, or ogling pretty patterns, I hope you'll enjoy.


[^1]: This isn't really a library in the original's sense, nor is it technically a 'port' rather it's a recreation in Apple's native frameworks rather than openFrameworks. Original library Copyright (c) 2008-2012 Memo Akten.