"""
PyTask Task Definitions for Structural Break Forecasting
=========================================================

Tasks for running Monte Carlo simulations and generating results.
Uses pytask v0.5+ syntax with Annotated types.
"""
from pathlib import Path
from typing import Annotated
import pickle
import sys

# Add project root to Python path so we can import our modules
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pytask import Product

# Define paths (task file is at project root)
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)


def task_run_variance_simulations(
    produces: Annotated[Path, Product] = RESULTS_DIR / "variance_mc_results.pkl"
) -> None:
    """Run variance break Monte Carlo simulations."""
    from analyses.simulations import mc_variance_breaks
    
    df_point, df_unc = mc_variance_breaks(
        n_sim=100,
        T=400,
        phi=0.6,
        window=100,
        horizon=20,
        seed=42
    )
    
    results = {
        "point_metrics": df_point,
        "uncertainty_metrics": df_unc
    }
    
    with open(produces, "wb") as f:
        pickle.dump(results, f)
    
    print(f"Saved variance MC results to {produces}")


def task_run_mean_simulations(
    produces: Annotated[Path, Product] = RESULTS_DIR / "mean_mc_results.pkl"
) -> None:
    """Run mean break Monte Carlo simulations."""
    from analyses.mean_simulations import mc_mean_breaks
    
    df_results = mc_mean_breaks(
        n_sim=100,
        T=300,
        Tb=150,
        window=60,
        seed=42
    )
    
    with open(produces, "wb") as f:
        pickle.dump(df_results, f)
    
    print(f"Saved mean MC results to {produces}")


def task_run_parameter_simulations(
    produces: Annotated[Path, Product] = RESULTS_DIR / "parameter_mc_results.pkl"
) -> None:
    """Run parameter break Monte Carlo simulations with Normal and Student-t innovations."""
    from analyses.param_simulations import mc_parameter_breaks_full
    
    all_err, df_results = mc_parameter_breaks_full(
        n_sim=100,
        T=400,
        Tb=200,
        t_post=250,
        window=80,
        innovations=[
            ("Gaussian", "normal", None),
            ("Student-t df=100", "student", 100),
            ("Student-t df=50", "student", 50),
        ],
        seed=42
    )
    
    results = {
        "errors": all_err,
        "summary": df_results
    }
    
    with open(produces, "wb") as f:
        pickle.dump(results, f)
    
    print(f"Saved parameter MC results to {produces}")


def task_plot_variance_logscore(
    depends_on: Path = RESULTS_DIR / "variance_mc_results.pkl",
    produces: Annotated[Path, Product] = FIGURES_DIR / "variance_logscore_comparison.png"
) -> None:
    """Generate variance logscore comparison plot."""
    import matplotlib
    matplotlib.use("Agg")  # Use non-interactive backend
    
    from analyses.plots import plot_logscore_comparison
    import matplotlib.pyplot as plt
    
    # plot_logscore_comparison saves to its own location, we'll copy
    plot_logscore_comparison()
    
    # Move the generated file to the correct location
    import shutil
    src_file = Path("variance_logscore_comparison.png")
    if src_file.exists():
        shutil.move(str(src_file), str(produces))
    
    plt.close("all")
    print(f"Saved variance plot to {produces}")


def task_plot_mean_results(
    depends_on: Path = RESULTS_DIR / "mean_mc_results.pkl",
    produces: Annotated[Path, Product] = FIGURES_DIR / "mean_results_bar.png"
) -> None:
    """Generate mean break results bar plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pickle
    
    from analyses.plots_mean import mean_plot_results_bar
    
    with open(depends_on, "rb") as f:
        df_results = pickle.load(f)
    
    mean_plot_results_bar(df_results, save_path=str(produces))
    plt.close("all")
    print(f"Saved mean plot to {produces}")


def task_plot_parameter_results(
    depends_on: Path = RESULTS_DIR / "parameter_mc_results.pkl",
    produces: Annotated[Path, Product] = FIGURES_DIR / "parameter_rmse_by_innovation.png"
) -> None:
    """Generate parameter break RMSE by innovation plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pickle
    
    from analyses.plots_parameter import param_plot_rmse_by_innovation
    
    with open(depends_on, "rb") as f:
        results = pickle.load(f)
    
    param_plot_rmse_by_innovation(results["summary"], save_path=str(produces))
    plt.close("all")
    print(f"Saved parameter plot to {produces}")
