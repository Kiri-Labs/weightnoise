"""Formatted reporting for weight noise analysis."""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
import os

console = Console()


def print_report(results):
    """Print a complete noise analysis report."""
    summary = results.get("summary", {})
    
    # Title
    console.print()
    console.print(Panel(
        f"[bold cyan]Weight Noise Report: {summary.get('model', '?')}[/]",
        box=box.ROUNDED,
    ))
    
    # Summary stats
    stats_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    stats_table.add_column("Metric", style="yellow")
    stats_table.add_column("Value", style="white")
    stats_table.add_row("Total parameters", f"{summary.get('total_params_m', '?'):.1f}M")
    stats_table.add_row("Estimated signal", f"{summary.get('estimated_signal_m', '?'):.1f}M")
    stats_table.add_row("Estimated noise", f"{summary.get('estimated_noise_m', '?'):.1f}M")
    stats_table.add_row("Noise percentage", f"[bold red]{summary.get('noise_percentage', '?'):.1f}%[/]")
    stats_table.add_row("Spectrally compressible", f"{summary.get('compressible_percentage', '?'):.1f}%")
    stats_table.add_row("Layers analyzed", str(summary.get('num_layers', '?')))
    stats_table.add_row("Weight matrices", str(summary.get('num_matrices', '?')))
    stats_table.add_row("Method", summary.get('method_used', 'spectral_noise_floor'))
    console.print(stats_table)
    console.print()

    # Per-layer table
    layer_table = Table(
        title="Layer-by-Layer Noise Analysis",
        box=box.MINIMAL_HEAVY_HEAD,
        header_style="bold cyan",
    )
    layer_table.add_column("Layer", justify="right", style="dim")
    layer_table.add_column("Mats", justify="right")
    layer_table.add_column("Params (M)", justify="right")
    layer_table.add_column("Mean |w|", justify="right")
    layer_table.add_column("Std", justify="right")
    layer_table.add_column("SNR", justify="right")
    layer_table.add_column("Noise SVs %", justify="right")
    layer_table.add_column("Compress %", justify="right")
    layer_table.add_column("Signal vs Noise", justify="center")

    for lidx in sorted(results.keys()):
        if lidx == "summary":
            continue
        layer = results[lidx]
        mats = list(layer["matrices"].values())
        if not mats:
            continue

        avg_snr = np.mean([m.get("snr", 0) for m in mats])
        avg_noise_sv = np.mean([m.get("noise_svals_ratio", 0) for m in mats])
        avg_compress = np.mean([m.get("compressibility_99pct", 0) for m in mats])
        avg_mean_abs = np.mean([abs(m.get("mean", 0)) for m in mats])
        avg_std = np.mean([m.get("std", 0) for m in mats])
        total_params = sum(m.get("param_count", 0) for m in mats)
        noise_pct = 100 * layer.get("noise_params", 0) / max(layer.get("total_params", 1), 1)

        # Signal bar
        signal_pct = 100 - noise_pct
        bar_len = 10
        signal_bars = int(signal_pct / 10)
        noise_bars = bar_len - signal_bars
        bar = f"[green]{'|' * signal_bars}[/][red]{'|' * noise_bars}[/]"

        layer_table.add_row(
            str(lidx),
            str(len(mats)),
            f"{total_params / 1e6:.2f}",
            f"{avg_mean_abs:.5f}",
            f"{avg_std:.5f}",
            f"{avg_snr:.2f}",
            f"{avg_noise_sv:.1f}",
            f"{avg_compress:.0f}",
            bar,
        )

    console.print(layer_table)

    # Overall assessment
    noise_pct = summary.get("noise_percentage", 0)
    if noise_pct > 70:
        assessment = "[bold red]HIGH NOISE — high compression potential[/]"
    elif noise_pct > 40:
        assessment = "[bold yellow]MODERATE NOISE — moderate compression possible[/]"
    else:
        assessment = "[bold green]LOW NOISE — most weights carry signal[/]"

    console.print()
    console.print(Panel(
        f"Assessment: {assessment}\n"
        f"Estimated size after noise removal: ~{summary.get('estimated_signal_m', 0):.0f}M params "
        f"(from {summary.get('total_params_m', 0):.0f}M)",
        box=box.ROUNDED,
    ))
    console.print()


def print_layer_detail(results, layer_idx):
    """Print detailed table for a single layer."""
    if layer_idx not in results:
        console.print(f"[red]Layer {layer_idx} not found[/]")
        return

    layer = results[layer_idx]
    table = Table(
        title=f"Layer {layer_idx} — Weight Matrix Detail",
        box=box.MINIMAL_HEAVY_HEAD,
        header_style="bold cyan",
    )
    table.add_column("Matrix", style="dim", no_wrap=True)
    table.add_column("Shape")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Skew", justify="right")
    table.add_column("Kurtosis", justify="right")
    table.add_column("SNR", justify="right")
    table.add_column("KL(Gauss)", justify="right")
    table.add_column("Noise SV%", justify="right")
    table.add_column("Energy%", justify="right")
    table.add_column("Compress%", justify="right")

    for name, m in sorted(layer["matrices"].items(), key=lambda x: x[0]):
        short = ".".join(name.split(".")[-3:]) if len(name) > 40 else name
        table.add_row(
            short,
            f"{m['shape'][0]}×{m['shape'][1]}",
            f"{m['mean']:.5f}",
            f"{m['std']:.5f}",
            f"{m['skew']:.2f}",
            f"{m['kurtosis']:.2f}",
            f"{m['snr']:.3f}",
            f"{m['kl_div_gaussian']:.3f}",
            f"{m['noise_svals_ratio']:.0f}",
            f"{m['noise_energy_pct']:.0f}",
            f"{m['compressibility_99pct']:.0f}",
        )

    console.print(table)

    # Summary for this layer
    total_noise = layer.get("noise_params", 0)
    total_params = layer.get("total_params", 1)
    console.print(f"\n  Layer {layer_idx}: {total_noise / 1e3:.0f}K noise params "
                  f"out of {total_params / 1e3:.0f}K total "
                  f"({100 * total_noise / total_params:.1f}%)")


try:
    import numpy as np
except ImportError:
    np = None
