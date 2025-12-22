# QUICKSTART - The Ultimate Acorn v7

Get up and running in 60 seconds!

## Installation (30 seconds)

```bash
# Install dependencies
pip install numpy Pillow

# Optional: pygame for future 2D client
pip install pygame
```

## Run Tests (15 seconds)

```bash
python run_tests.py
```

Expected output:
```
Total: 19 tests
Passed: 18 (94.7%)
✅ PRODUCTION READY
```

## Try It Out (15 seconds)

### Option 1: Fractal Demo
```bash
python main.py fractal
```

You'll see:
- 3 base entities created
- 6 fractal layers spawned (depth 1 and 2)
- Simulation running with statistics

### Option 2: Terminal Client
```bash
python main.py terminal
```

Commands to try:
```
look        # See your surroundings
move n      # Move north
stats       # View statistics
spawn basic 10 10    # Create entity
fractal spawn        # Create fractal layer
save test.png        # Save holographic plate
quit        # Exit
```

### Option 3: Headless Simulation
```bash
python main.py headless --ticks 100 --size 30
```

Creates a 30×30 world, runs 100 ticks, saves final state to PNG.

## What to Read Next

1. **README.md** - Full documentation
2. **FRACTAL_EXPERIMENTS.md** - How to use fractal features
3. **TEST_RESULTS.md** - Test coverage details
4. **DELIVERY_COMPLETE.md** - Complete system overview

## One-Minute Fractal Experiment

```python
# Create and run in Python REPL
from acorn import *
from acorn.fractal import FractalSimulationEngine

# Create universe
world = WorldState(30, 30)
config = {"fractal": {"enabled": True, "max_depth": 3}}
engine = AcornEngine(world, config)
fractal = FractalSimulationEngine(engine, config)

# Create entity
entity_id = engine.create_entity(EntityType.BASIC, Position(15, 15))

# Spawn fractal layer
layer_id = fractal.spawn_layer(entity_id, "test")

# Run it
for i in range(10):
    fractal.execute_layer(layer_id)
    print(f"Tick {i}: {fractal.layers[layer_id].tick_count} executed")

# See results
print(fractal.get_layer_hierarchy())
```

## That's It!

You now have a working fractal universe simulator.

**Explore. Experiment. Enjoy!** 🌰🌀
