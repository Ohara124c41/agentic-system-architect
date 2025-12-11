#!/usr/bin/env python3
"""
COGENT Trade-off Analysis Demo

Demonstrates the COGENT-style subsystem trade-off analysis for architecture decisions.
Based on the COGENT framework [Ohara, 2021] - Concurrent Generative Engineering.

This demo shows:
1. Subsystem stakeholder profiles with conflicting priorities
2. Technology scoring across optimization dimensions
3. Conflict identification between subsystems
4. Pareto frontier analysis
5. Resolution strategies

Usage:
    python scripts/demo_cogent.py
"""

import sys
import io
from pathlib import Path

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.cogent_tradeoffs import (
    SubsystemTradeoffAnalyzer,
    OptimizationDimension,
    ARCHITECTURE_SUBSYSTEMS,
    DIMENSION_CORRELATIONS,
)
from src.tools.knowledge_base import load_technology_profiles, get_technology_profile

# Optional matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Color scheme (matching visualize_results.py)
COLORS = {
    'primary': '#009999',      # Base teal
    'secondary': '#006666',    # Darker teal
    'accent': '#00CCCC',       # Lighter teal
    'complement1': '#994D00',  # Orange complement
    'complement2': '#990099',  # Purple complement
}


def generate_pareto_visualization(technologies: dict, analyzer, output_dir: str = "outputs/figures"):
    """
    Generate Pareto frontier visualization showing dominated vs non-dominated technologies.

    Creates a scatter plot for Cost vs Performance with:
    - Pareto frontier line connecting non-dominated points
    - Dominated points shown differently
    - Clear labeling for each technology
    """
    if not MATPLOTLIB_AVAILABLE:
        print("\n[Note: Install matplotlib to generate Pareto frontier visualization]")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ===== Subplot 1: Cost vs Performance =====
    ax1 = axes[0]
    dim_x, dim_y = OptimizationDimension.COST, OptimizationDimension.PERFORMANCE

    # Get Pareto frontier technologies
    pareto_techs = set(analyzer.get_pareto_frontier(dim_x, dim_y))

    # Plot all technologies
    for tech, scores in technologies.items():
        x, y = scores[dim_x], scores[dim_y]
        is_pareto = tech in pareto_techs

        color = COLORS['primary'] if is_pareto else COLORS['complement1']
        marker = 'o' if is_pareto else 's'
        size = 150 if is_pareto else 100

        ax1.scatter(x, y, c=color, s=size, marker=marker, edgecolors=COLORS['secondary'],
                   linewidth=2, zorder=3, label=tech)
        ax1.annotate(tech, (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold' if is_pareto else 'normal')

    # Draw Pareto frontier line
    pareto_points = [(scores[dim_x], scores[dim_y], tech)
                     for tech, scores in technologies.items() if tech in pareto_techs]
    pareto_points.sort(key=lambda p: p[0])  # Sort by x

    if len(pareto_points) >= 2:
        xs = [p[0] for p in pareto_points]
        ys = [p[1] for p in pareto_points]
        ax1.plot(xs, ys, '--', color=COLORS['primary'], linewidth=2, alpha=0.7, zorder=2)
        ax1.fill_between(xs, ys, [1.0]*len(ys), alpha=0.1, color=COLORS['primary'])

    ax1.set_xlabel('Cost (higher = cheaper)', fontsize=10)
    ax1.set_ylabel('Performance (higher = faster)', fontsize=10)
    ax1.set_title('Pareto Frontier: Cost vs Performance', fontsize=12, fontweight='bold',
                  color=COLORS['secondary'])
    ax1.set_xlim(0.4, 1.0)
    ax1.set_ylim(0.6, 1.0)
    ax1.grid(True, alpha=0.3)

    # ===== Subplot 2: Scalability vs Maintainability =====
    ax2 = axes[1]
    dim_x2, dim_y2 = OptimizationDimension.SCALABILITY, OptimizationDimension.MAINTAINABILITY

    pareto_techs2 = set(analyzer.get_pareto_frontier(dim_x2, dim_y2))

    for tech, scores in technologies.items():
        x, y = scores[dim_x2], scores[dim_y2]
        is_pareto = tech in pareto_techs2

        color = COLORS['primary'] if is_pareto else COLORS['complement1']
        marker = 'o' if is_pareto else 's'
        size = 150 if is_pareto else 100

        ax2.scatter(x, y, c=color, s=size, marker=marker, edgecolors=COLORS['secondary'],
                   linewidth=2, zorder=3)
        ax2.annotate(tech, (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold' if is_pareto else 'normal')

    # Draw Pareto frontier line
    pareto_points2 = [(scores[dim_x2], scores[dim_y2], tech)
                      for tech, scores in technologies.items() if tech in pareto_techs2]
    pareto_points2.sort(key=lambda p: p[0])

    if len(pareto_points2) >= 2:
        xs = [p[0] for p in pareto_points2]
        ys = [p[1] for p in pareto_points2]
        ax2.plot(xs, ys, '--', color=COLORS['primary'], linewidth=2, alpha=0.7, zorder=2)
        ax2.fill_between(xs, ys, [1.0]*len(ys), alpha=0.1, color=COLORS['primary'])

    ax2.set_xlabel('Scalability (higher = more scalable)', fontsize=10)
    ax2.set_ylabel('Maintainability (higher = easier to maintain)', fontsize=10)
    ax2.set_title('Pareto Frontier: Scalability vs Maintainability', fontsize=12, fontweight='bold',
                  color=COLORS['secondary'])
    ax2.set_xlim(0.4, 1.0)
    ax2.set_ylim(0.4, 0.9)
    ax2.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['primary'],
               markersize=10, label='Pareto Optimal', markeredgecolor=COLORS['secondary']),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['complement1'],
               markersize=10, label='Dominated', markeredgecolor=COLORS['secondary']),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle('Pareto Frontier Analysis\n(Non-dominated Technology Trade-offs)',
                 fontsize=14, fontweight='bold', color=COLORS['secondary'], y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    output_path = os.path.join(output_dir, 'pareto_frontier.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n[Pareto frontier visualization saved to: {output_path}]")


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str):
    """Print a section header."""
    print(f"\n--- {title} ---\n")


def demo_subsystem_profiles():
    """Demonstrate the subsystem stakeholder profiles."""
    print_header("1. SUBSYSTEM STAKEHOLDER PROFILES")
    print("""
The COGENT framework models software architecture decisions as a multi-stakeholder
optimization problem. Each subsystem (team/role) has different priorities:
""")

    print(f"{'Subsystem':<20} {'Owner':<20} {'Primary Priority':<20} {'Secondary'}")
    print("-" * 70)

    for sub_id, sub in ARCHITECTURE_SUBSYSTEMS.items():
        secondary = ", ".join([d.value for d in sub.secondary_dimensions])
        print(f"{sub.subsystem_name:<20} {sub.owner_role:<20} {sub.primary_dimension.value:<20} {secondary}")


def demo_dimension_correlations():
    """Demonstrate the dimension correlation matrix (conflicts)."""
    print_header("2. OPTIMIZATION DIMENSION CORRELATIONS")
    print("""
Some optimization dimensions conflict with each other. A negative correlation
means improving one dimension typically degrades the other:
""")

    print(f"{'Dimension A':<20} {'Dimension B':<20} {'Correlation':<15} {'Meaning'}")
    print("-" * 70)

    meanings = {
        -0.7: "Strong conflict",
        -0.6: "Moderate conflict",
        -0.5: "Moderate conflict",
        -0.4: "Mild conflict",
        -0.3: "Mild conflict",
    }

    for (dim_a, dim_b), corr in DIMENSION_CORRELATIONS.items():
        meaning = meanings.get(corr, "Unknown")
        print(f"{dim_a.value:<20} {dim_b.value:<20} {corr:<15} {meaning}")


def demo_technology_analysis():
    """Demonstrate technology analysis for different use cases."""
    print_header("3. TECHNOLOGY ANALYSIS: PostgreSQL vs MongoDB")

    analyzer = SubsystemTradeoffAnalyzer()

    # Use Case 1: Financial reporting system (favors PostgreSQL)
    print_section("Use Case: Financial Reporting System")
    print("Requirements: tabular_data, complex_joins, acid_transactions, reporting_systems")

    requirements = ["tabular_data", "complex_joins", "acid_transactions", "reporting_systems"]

    # Analyze PostgreSQL
    pg_profile = get_technology_profile("postgresql")
    if pg_profile:
        print("\n** PostgreSQL Analysis **")
        pg_analysis = analyzer.analyze_technology_for_subsystems("PostgreSQL", pg_profile, requirements)
        for sub_id, analysis in pg_analysis.items():
            icon = "✅" if analysis["satisfied"] else "⚠️"
            print(f"  {icon} {analysis['subsystem_name']}: {analysis['overall_score']:.0%}")
            if analysis["concerns"]:
                print(f"     Concerns: {', '.join(analysis['concerns'])}")

    # Analyze MongoDB
    mongo_profile = get_technology_profile("mongodb")
    if mongo_profile:
        print("\n** MongoDB Analysis **")
        mongo_analysis = analyzer.analyze_technology_for_subsystems("MongoDB", mongo_profile, requirements)
        for sub_id, analysis in mongo_analysis.items():
            icon = "✅" if analysis["satisfied"] else "⚠️"
            print(f"  {icon} {analysis['subsystem_name']}: {analysis['overall_score']:.0%}")
            if analysis["concerns"]:
                print(f"     Concerns: {', '.join(analysis['concerns'])}")


def demo_conflict_analysis():
    """Demonstrate conflict identification between subsystems."""
    print_header("4. SUBSYSTEM CONFLICT ANALYSIS")
    print("""
When subsystems have conflicting priorities, the orchestrator must mediate.
These conflicts are identified by analyzing dimension correlations:
""")

    analyzer = SubsystemTradeoffAnalyzer()
    conflicts = analyzer.identify_conflicts()

    if conflicts:
        for conflict in conflicts:
            print(f"\n⚡ {conflict.subsystem_a} ↔ {conflict.subsystem_b}")
            print(f"   Severity: {'🔴' * int(conflict.conflict_severity * 5)} ({conflict.conflict_severity:.0%})")
            dims = [f"{d[0].value} vs {d[1].value}" for d in conflict.conflicting_dimensions]
            print(f"   Conflicting Dimensions: {', '.join(dims)}")
            print(f"   Resolution Strategy: {conflict.resolution_strategy}")
    else:
        print("No significant conflicts identified.")


def demo_pareto_frontier():
    """Demonstrate Pareto frontier analysis."""
    print_header("5. PARETO FRONTIER ANALYSIS")
    print("""
The Pareto frontier shows technologies that are not dominated in any dimension.
A technology is dominated if another is better in all dimensions.
""")

    analyzer = SubsystemTradeoffAnalyzer()

    # Add technology scores (from visualization)
    technologies = {
        "PostgreSQL": {
            OptimizationDimension.PERFORMANCE: 0.7,
            OptimizationDimension.COST: 0.9,
            OptimizationDimension.SCALABILITY: 0.5,
            OptimizationDimension.MAINTAINABILITY: 0.8,
            OptimizationDimension.SECURITY: 0.8,
            OptimizationDimension.RELIABILITY: 0.9,
        },
        "MongoDB": {
            OptimizationDimension.PERFORMANCE: 0.8,
            OptimizationDimension.COST: 0.7,
            OptimizationDimension.SCALABILITY: 0.9,
            OptimizationDimension.MAINTAINABILITY: 0.6,
            OptimizationDimension.SECURITY: 0.7,
            OptimizationDimension.RELIABILITY: 0.7,
        },
        "Neo4j": {
            OptimizationDimension.PERFORMANCE: 0.8,
            OptimizationDimension.COST: 0.6,
            OptimizationDimension.SCALABILITY: 0.6,
            OptimizationDimension.MAINTAINABILITY: 0.5,
            OptimizationDimension.SECURITY: 0.7,
            OptimizationDimension.RELIABILITY: 0.7,
        },
        "Redis": {
            OptimizationDimension.PERFORMANCE: 0.95,
            OptimizationDimension.COST: 0.8,
            OptimizationDimension.SCALABILITY: 0.8,
            OptimizationDimension.MAINTAINABILITY: 0.7,
            OptimizationDimension.SECURITY: 0.6,
            OptimizationDimension.RELIABILITY: 0.6,
        },
    }

    for tech, scores in technologies.items():
        analyzer.add_technology_scores(tech, scores)

    # Show scores
    print_section("Technology Scores by Dimension")
    print(f"{'Technology':<15}", end="")
    dims = [OptimizationDimension.PERFORMANCE, OptimizationDimension.COST,
            OptimizationDimension.SCALABILITY, OptimizationDimension.MAINTAINABILITY]
    for d in dims:
        print(f"{d.value:<15}", end="")
    print()
    print("-" * 75)

    for tech, scores in technologies.items():
        print(f"{tech:<15}", end="")
        for d in dims:
            print(f"{scores[d]:<15.0%}", end="")
        print()

    # Find Pareto frontier
    print_section("Pareto Frontier: Cost vs Performance")
    pareto_cost_perf = analyzer.get_pareto_frontier(
        OptimizationDimension.COST,
        OptimizationDimension.PERFORMANCE
    )
    print(f"Technologies on frontier: {', '.join(pareto_cost_perf)}")
    print("(These technologies offer the best trade-off between cost and performance)")

    print_section("Pareto Frontier: Scalability vs Maintainability")
    pareto_scale_maint = analyzer.get_pareto_frontier(
        OptimizationDimension.SCALABILITY,
        OptimizationDimension.MAINTAINABILITY
    )
    print(f"Technologies on frontier: {', '.join(pareto_scale_maint)}")
    print("(These technologies offer the best trade-off between scalability and maintainability)")

    # Generate Pareto visualization if matplotlib is available
    generate_pareto_visualization(technologies, analyzer)


def demo_full_report():
    """Generate a full COGENT report for a technology."""
    print_header("6. FULL COGENT REPORT: Docker Compose")

    analyzer = SubsystemTradeoffAnalyzer()
    profile = get_technology_profile("docker_compose")
    requirements = ["small_deployments", "limited_ops_experience", "mvp_stage", "rapid_deployment"]

    if profile:
        subsystem_analysis = analyzer.analyze_technology_for_subsystems(
            "Docker Compose", profile, requirements
        )
        conflicts = analyzer.identify_conflicts()

        report = analyzer.format_tradeoff_report("Docker Compose", subsystem_analysis, conflicts)
        print(report)


def main():
    """Run the COGENT demo."""
    print("\n" + "=" * 70)
    print("  COGENT-STYLE SUBSYSTEM TRADE-OFF ANALYSIS DEMO")
    print("  Based on COGENT Framework [Ohara, 2021]")
    print("=" * 70)
    print("""
This demo showcases the COGENT (Concurrent Generative Engineering) approach
to multi-stakeholder architecture decision-making. The system models:

- Subsystem stakeholders with different optimization priorities
- Dimension correlations that identify conflicts
- Pareto-optimal solutions across trade-off dimensions
- Resolution strategies for conflicting requirements

Originally developed for concurrent engineering of electro-mechanical systems,
this framework is now applied to software architecture technology selection.
""")

    demo_subsystem_profiles()
    demo_dimension_correlations()
    demo_technology_analysis()
    demo_conflict_analysis()
    demo_pareto_frontier()
    demo_full_report()

    print_header("DEMO COMPLETE")
    print("""
This analysis demonstrates how COGENT helps architects:
1. Identify stakeholder conflicts BEFORE they become problems
2. Quantify trade-offs between competing priorities
3. Find Pareto-optimal solutions that balance constraints
4. Communicate resolution strategies to stakeholders

For more information, see:
- src/features/cogent_tradeoffs.py (implementation)
- docs/methods.md Section 16.6 (documentation)
""")


if __name__ == "__main__":
    main()
