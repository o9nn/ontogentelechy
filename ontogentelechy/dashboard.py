"""
Ontogentelechy Interactive Dashboard

Streamlit app for real-time visualization of:
- Actualization trajectories
- Attractor landscapes (PCA)
- Phase transition timelines
- Benchmark results

Run with:
    streamlit run -m ontogentelechy.dashboard
    OR
    python -m ontogentelechy.dashboard
"""

from __future__ import annotations
import sys


def _check_streamlit() -> bool:
    try:
        import streamlit  # noqa: F401
        return True
    except ImportError:
        return False


def _check_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def run_dashboard() -> None:
    """Launch the Streamlit dashboard."""
    if not _check_streamlit():
        print("Streamlit is required. Install with: pip install ontogentelechy[visualization]")
        sys.exit(1)
    
    import streamlit as st
    import numpy as np
    
    has_matplotlib = _check_matplotlib()
    if has_matplotlib:
        import matplotlib.pyplot as plt
    
    from .core import ActualizationTracker
    from .entity import SimpleEntity
    from .benchmarks import (
        CoherentClusterBenchmark,
        PhaseSeparationBenchmark,
        EmergenceTrackingBenchmark,
        BenchmarkResult,
    )
    from .examples import EXAMPLE_TELOI
    
    st.set_page_config(
        page_title="Ontogentelechy Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("🎯 Ontogentelechy Dashboard")
    st.markdown("*Purpose-Driven Cognitive Development Framework*")
    
    # Sidebar
    st.sidebar.header("Configuration")
    mode = st.sidebar.selectbox(
        "Mode",
        ["Live Tracking", "Benchmark", "Attractor Landscape"],
    )
    
    telos_name = st.sidebar.selectbox(
        "Telos",
        list(EXAMPLE_TELOI.keys()),
    )
    telos = EXAMPLE_TELOI[telos_name]()
    
    if mode == "Live Tracking":
        st.header("Live Actualization Tracking")
        
        dim = st.sidebar.slider("State dimension", 2, 50, 10)
        n_steps = st.sidebar.slider("Steps", 10, 200, 50)
        
        if st.button("Run Simulation"):
            tracker = ActualizationTracker()
            state = np.random.uniform(0.1, 0.3, dim)
            entity = SimpleEntity(state=state)
            
            actualization_vals = []
            emergence_vals = []
            integration_vals = []
            phases = []
            
            progress_bar = st.progress(0)
            
            for step in range(n_steps):
                state += np.random.normal(0.02, 0.01, dim)
                state = np.clip(state, 0.0, 1.0)
                entity.update_state(state)
                
                metrics = tracker.compute_metrics(entity, telos)
                actualization_vals.append(metrics.actualization)
                emergence_vals.append(metrics.emergence)
                integration_vals.append(metrics.integration)
                phases.append(telos.phase.value)
                progress_bar.progress((step + 1) / n_steps)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Actualization", f"{actualization_vals[-1]:.3f}")
            col2.metric("Final Emergence", f"{emergence_vals[-1]:.3f}")
            col3.metric("Final Phase", phases[-1])
            
            if has_matplotlib:
                fig, axes = plt.subplots(2, 1, figsize=(10, 6))
                
                axes[0].plot(actualization_vals, label="Actualization", color="blue")
                axes[0].plot(emergence_vals, label="Emergence", color="orange")
                axes[0].plot(integration_vals, label="Integration", color="green")
                axes[0].set_title("Actualization Metrics Over Time")
                axes[0].legend()
                axes[0].set_ylim(0, 1)
                axes[0].grid(True, alpha=0.3)
                
                phase_colors = {
                    'potential': 'gray',
                    'emergent': 'blue',
                    'developing': 'green',
                    'actualizing': 'orange',
                    'actualized': 'gold',
                }
                phase_nums = [list(phase_colors.keys()).index(p) if p in phase_colors else 0 
                             for p in phases]
                axes[1].fill_between(range(len(phase_nums)), phase_nums, alpha=0.6, color='purple')
                axes[1].set_title("Phase Timeline")
                axes[1].set_yticks(range(5))
                axes[1].set_yticklabels(list(phase_colors.keys()))
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            
            st.subheader("Phase Transitions")
            st.write(f"Detected {len(tracker.phase_transitions)} phase transition(s)")
    
    elif mode == "Benchmark":
        st.header("Benchmark Suite")
        
        benchmark_name = st.sidebar.selectbox(
            "Benchmark",
            ["coherent_cluster", "phase_separation", "emergence_tracking"],
        )
        
        n_generations = st.sidebar.slider("Generations", 20, 200, 50)
        
        if st.button("Run Benchmark"):
            with st.spinner(f"Running {benchmark_name} benchmark..."):
                if benchmark_name == "coherent_cluster":
                    bench = CoherentClusterBenchmark(population_size=30)
                    result = bench.run(n_generations=n_generations)
                elif benchmark_name == "phase_separation":
                    bench = PhaseSeparationBenchmark(population_size=30)
                    result = bench.run(n_generations=n_generations)
                else:
                    bench = EmergenceTrackingBenchmark()
                    result = bench.run()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Fitness", f"{result.final_fitness:.4f}")
            col2.metric("Time (s)", f"{result.time_seconds:.2f}")
            conv_gen = result.convergence_generation
            col3.metric("Convergence Gen", str(conv_gen) if conv_gen else "N/A")
            
            st.text(result.summary())
            
            if has_matplotlib:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(result.mean_fitness_history, label="Mean Fitness", color="blue")
                ax.plot(result.max_fitness_history, label="Max Fitness", color="red")
                ax.set_title(f"{benchmark_name} — Fitness History")
                ax.set_xlabel("Generation")
                ax.set_ylabel("Fitness")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
    
    elif mode == "Attractor Landscape":
        st.header("Attractor Landscape Visualization")
        
        if not has_matplotlib:
            st.warning("matplotlib is required for landscape visualization.")
        else:
            from .attractor import AttractorLandscape, Basin, Repulsor
            
            n_basins = st.sidebar.slider("Number of basins", 1, 5, 2)
            n_repulsors = st.sidebar.slider("Number of repulsors", 0, 3, 1)
            dim = st.sidebar.slider("Dimension", 2, 20, 5)
            n_entities = st.sidebar.slider("Entity trajectory steps", 5, 50, 20)
            
            if st.button("Generate Landscape"):
                rng = np.random.RandomState(42)
                
                basins = [
                    Basin(
                        state=rng.uniform(0.2, 0.8, dim),
                        label=f"Attractor {i+1}",
                        strength=rng.uniform(0.5, 1.5),
                    )
                    for i in range(n_basins)
                ]
                repulsors = [
                    Repulsor(
                        state=rng.uniform(0.3, 0.7, dim),
                        label=f"Repulsor {i+1}",
                    )
                    for i in range(n_repulsors)
                ]
                landscape = AttractorLandscape(basins=basins, repulsors=repulsors)
                
                state = rng.uniform(0.1, 0.9, dim)
                trajectory = [state.copy()]
                for _ in range(n_entities - 1):
                    grad = landscape.net_gradient(state)
                    state = np.clip(state + grad * 0.1 + rng.normal(0, 0.02, dim), 0.0, 1.0)
                    trajectory.append(state.copy())
                
                try:
                    fig = landscape.visualize(entity_states=trajectory, title="Attractor Landscape (PCA)")
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Visualization error: {e}")


if __name__ == "__main__":
    run_dashboard()
