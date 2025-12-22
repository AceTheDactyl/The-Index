#!/usr/bin/env python3
"""
The Ultimate Acorn v7 - Main Launcher

Provides easy access to all modes:
- Terminal client
- Headless simulation
- Test suite
- Quick demos
"""

import sys
import argparse
import time


def run_terminal():
    """Launch terminal client."""
    from clients.terminal import main
    main()


def run_headless(ticks: int, world_size: int):
    """Run headless simulation."""
    from acorn import AcornEngine, WorldState, EntityType, Position
    from acorn.plates import PlateManager
    
    print(f"Starting headless simulation: {ticks} ticks, {world_size}x{world_size} world")
    
    world = WorldState(world_size, world_size, seed=int(time.time()))
    config = {
        "iss_enabled": True,
        "fractal": {
            "enabled": True,
            "max_depth": 3,
            "budget_decay": 0.5
        }
    }
    engine = AcornEngine(world, config)
    engine.initialize_iss()
    
    # Add some entities
    for i in range(10):
        x = (i * 7) % world_size
        y = (i * 11) % world_size
        engine.create_entity(EntityType.BASIC, Position(x, y))
    
    print(f"Created {len(world.entities)} entities")
    
    # Run simulation
    start_time = time.time()
    for i in range(ticks):
        result = engine.tick()
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            tps = (i + 1) / elapsed
            print(f"Tick {result['tick']}: {tps:.1f} TPS, {result['entities']} entities")
    
    duration = time.time() - start_time
    avg_tps = ticks / duration
    
    print(f"\nSimulation complete!")
    print(f"Total time: {duration:.2f}s")
    print(f"Average TPS: {avg_tps:.1f}")
    
    # Save final state
    manager = PlateManager(engine)
    filename = f"headless_final_{int(time.time())}.png"
    manager.save_plate(filename)
    print(f"Saved final state to {filename}")
    
    # Print stats
    stats = engine.get_stats()
    print("\nFinal statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def run_tests(verbose: bool):
    """Run test suite."""
    import run_tests
    # Tests will run via import and main()


def run_fractal_demo():
    """Run fractal simulation demo."""
    from acorn import AcornEngine, WorldState, EntityType, Position
    from acorn.fractal import FractalSimulationEngine
    
    print("=" * 60)
    print("FRACTAL SIMULATION DEMO")
    print("=" * 60)
    print()
    
    # Create base universe
    world = WorldState(30, 30, seed=42)
    config = {
        "iss_enabled": True,
        "fractal": {
            "enabled": True,
            "max_depth": 4,
            "budget_decay": 0.5,
            "base_budget": 50
        }
    }
    engine = AcornEngine(world, config)
    engine.initialize_iss()
    
    # Create fractal engine
    fractal = FractalSimulationEngine(engine, config)
    
    # Create base entities
    entity_ids = []
    for i in range(3):
        eid = engine.create_entity(EntityType.BASIC, Position(10 + i*5, 10 + i*5))
        entity_ids.append(eid)
    
    print(f"Created {len(entity_ids)} base entities")
    print()
    
    # Spawn fractal layers
    print("Spawning fractal layers...")
    layer_count = 0
    for entity_id in entity_ids:
        for depth in range(2):
            layer_id = fractal.spawn_layer(entity_id, f"demo_depth_{depth}")
            if layer_id:
                layer_count += 1
                print(f"  Layer {layer_id}: depth {depth+1}")
    
    print(f"\nTotal layers spawned: {layer_count}")
    print()
    
    # Run simulation
    print("Running fractal simulation...")
    for tick in range(20):
        # Base universe
        engine.tick()
        
        # All fractal layers
        results = fractal.execute_all_layers()
        
        if (tick + 1) % 5 == 0:
            hierarchy = fractal.get_layer_hierarchy()
            active_layers = sum(len(layers) for layers in hierarchy['base']['layers'].values())
            print(f"Tick {tick+1}: {active_layers} active layers")
    
    # Cleanup and show results
    fractal.cleanup_exhausted_layers()
    
    print()
    print("Fractal simulation complete!")
    print()
    
    # Show hierarchy
    hierarchy = fractal.get_layer_hierarchy()
    print("Final layer hierarchy:")
    print(f"  Base (depth 0): {hierarchy['base']['entities']} entities")
    for depth, layers in sorted(hierarchy['base']['layers'].items()):
        print(f"  Depth {depth}: {len(layers)} layers")
    
    print()
    
    # Show stats
    stats = fractal.get_stats()
    print("Fractal statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="The Ultimate Acorn v7 - Fractal Universe Simulator"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Terminal client
    subparsers.add_parser("terminal", help="Launch terminal client")
    
    # Headless simulation
    headless = subparsers.add_parser("headless", help="Run headless simulation")
    headless.add_argument("--ticks", type=int, default=1000, help="Number of ticks to run")
    headless.add_argument("--size", type=int, default=50, help="World size")
    
    # Tests
    tests = subparsers.add_parser("test", help="Run test suite")
    tests.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Fractal demo
    subparsers.add_parser("fractal", help="Run fractal simulation demo")
    
    # Parse args
    args = parser.parse_args()
    
    if not args.command:
        # No command, show help
        parser.print_help()
        print("\n" + "=" * 60)
        print("Quick start:")
        print("  python main.py terminal     - Interactive terminal interface")
        print("  python main.py fractal      - Fractal simulation demo")
        print("  python main.py test         - Run test suite")
        print("  python main.py headless     - Headless simulation")
        print("=" * 60)
        sys.exit(0)
    
    # Run command
    if args.command == "terminal":
        run_terminal()
    elif args.command == "headless":
        run_headless(args.ticks, args.size)
    elif args.command == "test":
        run_tests(args.verbose)
    elif args.command == "fractal":
        run_fractal_demo()


if __name__ == "__main__":
    main()
