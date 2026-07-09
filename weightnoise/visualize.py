"""Formatted reporting for weight noise analysis."""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import numpy as np

console = Console()


def print_report(results):
    """Print a complete noise analysis report."""
    summary = results.get("summary", {})

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
    noise_pct = summary.get('noise_percentage', 0)
    noise_str = f"[bold red]{noise_pct:.1f}%[/]" if noise_pct > 30 else f"[bold green]{noise_pct:.1f}%[/]"
    stats_table.add_row("Noise percentage", noise_str)
    stats_table.add_row("Layers analyzed", str(summary.get('num_layers', '?')))
    stats_table.add_row("Weight matrices", str(summary.get('num_matrices', '?')))
    stats_table.add_row("Method", "Wanda importance + Spectral analysis")
    console.print(stats_table)
    console.print()

    # Per-layer table
    layer_table = Table(
        title="Layer-by-Layer Spectral Noise Analysis",
        box=box.MINIMAL_HEAVY_HEAD,
        header_style="bold cyan",
    )
    layer_table.add_column("Layer", justify="right", style="dim")
    layer_table.add_column("Mats", justify="right")
    layer_table.add_column("Params (M)", justify="right")
    layer_table.add_column("Kurtosis", justify="right")
    layer_table.add_column("KL(Gauss)", justify="right")
    layer_table.add_column("EffRank%", justify="right")
    layer_table.add_column("Top10%", justify="right")
    layer_table.add_column("Low=Noise%", justify="right")
    layer_table.add_column("Signal vs Noise", justify="center")

    for lidx in sorted(k for k in results if isinstance(k, int)):
        layer = results[lidx]
        mats = list(layer["matrices"].values())
        if not mats:
            continue

        avg_kurt = np.mean([m.get("kurtosis", 0) for m in mats])
        avg_kl = np.mean([m.get("kl_div_gaussian", 0) for m in mats])
        avg_eff_rank = np.mean([m.get("eff_rank_pct", 0) for m in mats])
        avg_conc = np.mean([m.get("top10_concentration", 0) for m in mats])
        avg_lowimp = np.mean([m.get("low_importance_pct", 0) for m in mats])
        total_params = sum(m.get("param_count", 0) for m in mats)
        noise_pct = 100.0 * layer.get("noise_params", 0) / max(total_params, 1)

        signal_pct = 100.0 - noise_pct
        bar_len = 10
        signal_bars = int(signal_pct / 10)
        noise_bars = bar_len - signal_bars
        bar = f"[green]{'|' * signal_bars}[/][red]{'|' * noise_bars}[/]"

        layer_table.add_row(
            str(lidx),
            str(len(mats)),
            f"{total_params / 1e6:.2f}",
            f"{avg_kurt:.1f}",
            f"{avg_kl:.2f}",
            f"{avg_eff_rank:.0f}",
            f"{avg_conc:.0f}",
            f"{avg_lowimp:.0f}",
            bar,
        )

    console.print(layer_table)

    noise_pct = summary.get("noise_percentage", 0)
    if noise_pct > 50:
        assessment = f"[bold red]HIGH NOISE \u2014 {noise_pct:.0f}% of weights carry minimal signal[/]"
    elif noise_pct > 20:
        assessment = f"[bold yellow]MODERATE NOISE \u2014 {noise_pct:.0f}% of weights are low-importance[/]"
    else:
        assessment = f"[bold green]LOW NOISE \u2014 most weights carry signal[/]"

    console.print()
    console.print(Panel(
        f"Assessment: {assessment}\n"
        f"Estimated signal: {summary.get('estimated_signal_m', 0):.0f}M params "
        f"(from {summary.get('total_params_m', 0):.0f}M total)",
        box=box.ROUNDED,
    ))
    console.print()


def print_layer_detail(results, layer_idx):
    """Print detailed table for a single layer."""
    key = int(layer_idx) if isinstance(layer_idx, str) else layer_idx
    if key not in results:
        console.print(f"[red]Layer {layer_idx} not found[/]")
        return

    layer = results[key]
    table = Table(
        title=f"Layer {key} \u2014 Weight Matrix Detail",
        box=box.MINIMAL_HEAVY_HEAD,
        header_style="bold cyan",
    )
    table.add_column("Matrix", style="dim", no_wrap=True)
    table.add_column("Shape")
    table.add_column("Kurtosis", justify="right")
    table.add_column("KL(Gauss)", justify="right")
    table.add_column("EffRank%", justify="right")
    table.add_column("Top10%", justify="right")
    table.add_column("Rank@99%", justify="right")
    table.add_column("NearZero%", justify="right")
    table.add_column("Noise%", justify="right")

    for name, m in sorted(layer["matrices"].items(), key=lambda x: x[0]):
        short = ".".join(name.split(".")[-3:]) if len(name) > 40 else name
        table.add_row(
            short,
            f"{m['shape'][0]}\u00d7{m['shape'][1]}",
            f"{m['kurtosis']:.1f}",
            f"{m['kl_div_gaussian']:.2f}",
            f"{m['eff_rank_pct']:.0f}",
            f"{m['top10_concentration']:.0f}",
            f"{m['rank_99_pct']:.0f}",
            f"{m['near_zero_pct']:.0f}",
            f"{m['low_importance_pct']:.1f}",
        )

    console.print(table)

    total_noise = layer.get("noise_params", 0)
    total_params = layer.get("total_params", 1)
    console.print(f"\n  Layer {key}: {total_noise / 1e3:.0f}K noise params "
                  f"out of {total_params / 1e3:.0f}K total "
                  f"({100.0 * total_noise / max(total_params, 1):.1f}%)")
