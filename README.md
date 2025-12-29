# Ontogentelechy

**Purpose-Driven Cognitive Development Framework**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## What is Ontogentelechy?

**Ontogentelechy** = **Ontogenesis** + **Teleology** + **Entelechy**

A self-organizing developmental framework that integrates:

- **Ontogenesis** (ὄντος + γένεσις) - Development and coming-into-being
- **Teleology** (τέλος + λογία) - Purpose-driven, goal-oriented development
- **Entelechy** (ἐντελέχεια) - Realization of potential, actualization of form

**Core Principle:** Systems develop with intrinsic purpose, moving toward actualization of their potential through stages of increasing complexity and integration.

---

## Key Features

### 🎯 Purpose-Driven Development
- Systems develop with **intrinsic purpose**, not just blind optimization
- Natural convergence toward actualized forms
- Meaningful cognitive structures

### 📊 Actualization Tracking
- **Potentiality** - Unrealized capabilities
- **Emergence** - New properties appearing
- **Integration** - Component coherence
- **Actualization** - Overall progress
- **Telos Alignment** - Alignment with purpose

### 🧲 Developmental Attractors
- Stable configurations toward which development naturally tends
- Gradient-based guidance toward purpose
- Basin of attraction detection

### 🔄 Phase Transitions
- Automatic detection of qualitative changes
- 5-phase progression: **potential** → **emergent** → **developing** → **actualizing** → **actualized**

### 🏆 Teleological Fitness
- Multi-component evaluation:
  - Base fitness (25%)
  - Telos alignment (30%)
  - Actualization progress (25%)
  - Emergent properties (20%)

---

## Installation

```bash
pip install ontogentelechy
```

Or install from source:

```bash
git clone https://github.com/o9nn/ontogentelechy.git
cd ontogentelechy
pip install -e .
```

---

## Quick Start

### Basic Usage

```python
from ontogentelechy import Telos, Criterion, ActualizationTracker
from ontogentelechy.examples import create_semantic_coherence_telos

# Create a telos (intrinsic purpose)
telos = create_semantic_coherence_telos()

# Create actualization tracker
tracker = ActualizationTracker()

# Track development
metrics = tracker.compute_metrics(entity, telos)
print(f"Actualization: {metrics.actualization:.3f}")
print(f"Phase: {telos.phase.value}")
```

### Custom Telos

```python
from ontogentelechy import Telos, Criterion

def evaluate_creativity(entity):
    # Custom evaluation logic
    return 0.8

# Define custom purpose
creative_telos = Telos(
    name="creative_emergence",
    description="Maximize creative novel emergence",
    actualization_criteria=[
        Criterion("novelty", "Novel patterns", 0.5, evaluate_creativity, 1.0),
        Criterion("coherence", "Pattern coherence", 0.3, evaluate_coherence, 1.0),
        Criterion("utility", "Practical utility", 0.2, evaluate_utility, 1.0)
    ],
    attractor_state={'creativity': 0.9, 'coherence': 0.7}
)
```

### Developmental Attractor

```python
from ontogentelechy import DevelopmentalAttractor
import numpy as np

# Create attractor
attractor = DevelopmentalAttractor(telos)

# Compute gradient toward purpose
current_state = np.array([0.5, 0.6, 0.7])
gradient = attractor.compute_gradient(current_state)

# Apply pull toward attractor
attractor.apply_pull(entity, strength=1.0)
```

---

## Example Teloi

The framework includes 5 pre-defined purposes:

1. **Semantic Coherence** - Achieve coherent semantic representation
2. **Adaptive Learning** - Continuously learn while maintaining stability
3. **Complexity Emergence** - Develop complex structures through emergence
4. **Efficient Computation** - Achieve efficiency while maintaining effectiveness
5. **Knowledge Integration** - Integrate diverse knowledge into coherent whole

```python
from ontogentelechy.examples import (
    create_semantic_coherence_telos,
    create_adaptive_learning_telos,
    create_complexity_emergence_telos,
    create_efficient_computation_telos,
    create_knowledge_integration_telos,
    list_example_teloi
)

# List all available teloi
print(list_example_teloi())

# Use a pre-defined telos
telos = create_adaptive_learning_telos()
```

---

## Architecture

```
ontogentelechy/
├── core.py              # Core framework
│   ├── Telos            # Intrinsic purpose
│   ├── Criterion        # Actualization measurement
│   ├── ActualizationPhase
│   ├── ActualizationMetrics
│   ├── ActualizationTracker
│   └── DevelopmentalAttractor
├── fitness.py           # Fitness functions
│   ├── TeleologicalFitness
│   └── MultiTelosFitness
└── examples.py          # Pre-defined teloi
```

---

## Philosophical Foundation

### Aristotelian Teleology
- **Final Cause** (τέλος) - The purpose for which something exists
- **Formal Cause** - The pattern being actualized
- **Efficient Cause** - The process of actualization
- **Material Cause** - The substrate undergoing development

### Process Philosophy
- **Becoming over Being** - Reality as process, not static
- **Creativity** - Novel emergence in development
- **Concrescence** - Growing together into unity
- **Prehension** - Grasping and integrating experience

### Enactivism
- **Autonomy** - Self-organizing and self-maintaining
- **Sense-making** - Creating meaning through interaction
- **Emergence** - New properties arising from interaction
- **Embodiment** - Cognition as embodied action

---

## Documentation

- [Design Document](docs/ONTOGENTELECHY_DESIGN.md) - Complete philosophical and technical design
- [Implementation Guide](docs/ONTOGENTELECHY_IMPLEMENTATION.md) - Implementation details and usage
- [Complete Overview](docs/ONTOGENTELECHY_COMPLETE.md) - Comprehensive overview

---

## Use Cases

### Cognitive Architecture Development
- Purpose-driven evolution of cognitive structures
- Meaningful development of knowledge representations
- Self-organizing semantic networks

### Machine Learning
- Fitness functions with intrinsic purpose
- Evolutionary algorithms with teleological guidance
- Developmental neural architectures

### Artificial Life
- Self-actualizing artificial organisms
- Purpose-driven morphogenesis
- Emergent complexity with meaning

### Multi-Agent Systems
- Collective actualization
- Emergent group purposes
- Coordinated development

---

## Requirements

- Python 3.9+
- NumPy
- (Optional) Integration with evolutionary frameworks

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Citation

If you use Ontogentelechy in your research, please cite:

```bibtex
@software{ontogentelechy2025,
  title = {Ontogentelechy: Purpose-Driven Cognitive Development Framework},
  author = {O9NN},
  year = {2025},
  url = {https://github.com/o9nn/ontogentelechy}
}
```

---

## Acknowledgments

**Philosophical Foundations:**
- Aristotle - Teleology and four causes
- Alfred North Whitehead - Process philosophy
- Francisco Varela - Enactivism and autopoiesis
- Hans Driesch - Entelechy concept

**Technical Inspirations:**
- OpenCog - Cognitive architecture
- MOSES - Meta-Optimizing Semantic Evolution
- Developmental systems theory
- Attractor dynamics in complex systems

---

## Contact

- **Repository:** https://github.com/o9nn/ontogentelechy
- **Issues:** https://github.com/o9nn/ontogentelechy/issues
- **Organization:** https://github.com/o9nn

---

*"The acorn doesn't become an oak by accident—it actualizes its intrinsic potential through purposeful development. So too should our cognitive architectures."*

---

**Status:** ✅ Production Ready  
**Version:** 0.1.0  
**Last Updated:** December 2025
