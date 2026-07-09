#!/usr/bin/env python3
"""weightnoise CLI — analyze, visualize, and remove noise from neural network weights."""


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="weightnoise",
        description="Visualize and remove noise from neural network weights.\n"
        "Analyze any HuggingFace model's noise profile, generate compression plans,\n"
        "prune low-importance weights, or compare two models side by side.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── inspect ──
    inspect_p = sub.add_parser("inspect", help="Inspect model weights for noise")
    inspect_p.add_argument("model", help="HF model ID or local path")
    inspect_p.add_argument("--layer", type=int, default=None, help="Single layer to inspect")
    inspect_p.add_argument("--threshold", type=float, default=0.01,
                           help="Noise threshold (fraction of max singular value)")
    inspect_p.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    inspect_p.add_argument("--trust-remote-code", action="store_true",
                           help="Trust remote code for custom architectures")

    # ── plan ──
    plan_p = sub.add_parser("plan", help="Generate compression plan from noise analysis")
    plan_p.add_argument("model", help="HF model ID or local path")
    plan_p.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    plan_p.add_argument("--trust-remote-code", action="store_true",
                        help="Trust remote code for custom architectures")

    # ── compare ──
    compare_p = sub.add_parser("compare", help="Compare noise profiles of two models")
    compare_p.add_argument("model_a", help="First model")
    compare_p.add_argument("model_b", help="Second model")
    compare_p.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    compare_p.add_argument("--trust-remote-code", action="store_true",
                           help="Trust remote code for custom architectures")

    # ── prune ──
    prune_p = sub.add_parser("prune", help="Remove noise from model weights")
    prune_p.add_argument("model", help="HF model ID or local path")
    prune_p.add_argument("--output", "-o", default=None, help="Output path for pruned model")
    prune_p.add_argument("--keep", type=float, default=0.5,
                         help="Fraction of weights to keep (1.0 = none removed)")
    prune_p.add_argument("--method", default="wanda",
                         choices=["magnitude", "spectral", "wanda"],
                         help="Pruning method. wanda = weight×activation norm (recommended)")
    prune_p.add_argument("--threshold", type=float, default=0.01,
                         help="Noise threshold for spectral method")
    prune_p.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    prune_p.add_argument("--trust-remote-code", action="store_true",
                         help="Trust remote code for custom architectures")
    prune_p.add_argument("--save", action="store_true",
                         help="Save pruned model to disk")

    args = parser.parse_args()

    # ── COMMAND: inspect ──
    if args.command == "inspect":
        from .inspect import NoiseInspector
        from .visualize import print_report, print_layer_detail

        inspector = NoiseInspector(args.model, device=args.device,
                                   trust_remote_code=args.trust_remote_code)
        results = inspector.analyze(threshold=args.threshold)

        if args.layer is not None:
            print_layer_detail(results, args.layer)
        else:
            print_report(results)

        summary = results["summary"]
        print(f"\n  Total: {summary['total_params_m']:.0f}M params")
        print(f"  Noise: {summary['noise_percentage']:.1f}% ({summary['estimated_noise_m']:.0f}M)")
        print(f"  Signal: {summary['estimated_signal_m']:.0f}M")
        print(f"  Run \033[1mweightnoise plan {args.model}\033[0m for compression recommendations")

    # ── COMMAND: plan ──
    elif args.command == "plan":
        from .inspect import NoiseInspector
        from .planner import generate_plan
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box

        console = Console()
        inspector = NoiseInspector(args.model, device=args.device,
                                   trust_remote_code=args.trust_remote_code)
        results = inspector.analyze()
        plan = generate_plan(results)

        # Title
        console.print()
        console.print(Panel(
            f"[bold cyan]Compression Plan: {args.model}[/]",
            box=box.ROUNDED,
        ))

        # Summary
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        t.add_column("Metric", style="yellow")
        t.add_column("Value")
        p = plan["plan"]
        t.add_row("Total parameters", f"{p['compressed_m']:.0f}M (from {plan['total_m']:.0f}M)")
        t.add_row("Compression ratio", f"{p['compression_ratio']:.1f}x")
        t.add_row("Prunable noise", f"{p['prunable_params_m']:.1f}M")
        t.add_row("SVD-compressible", f"{p['svdable_params_m']:.1f}M")
        t.add_row("Noise percentage", f"{plan['noise_pct']:.1f}%")
        t.add_row("Layers in plan", str(p["layer_count"]))
        console.print(t)

        # Hardware targets
        console.print("\n[bold]Hardware Targets:[/]")
        hw_table = Table(box=box.MINIMAL_HEAVY_HEAD, header_style="bold cyan")
        hw_table.add_column("Hardware")
        hw_table.add_column("VRAM", justify="right")
        hw_table.add_column("Fits fp16?", justify="center")
        hw_table.add_column("Compressed", justify="right")

        for key, hw in plan["hardware_targets"].items():
            if key in ["rtx_4090", "rtx_4080", "rtx_3080", "mac_m1", "colab_free"]:
                fit_str = "[green]YES[/]" if hw["fits_fp16"] else f"[red]NO[/] ({hw['needs_quant']})"
                hw_table.add_row(hw["name"], f"{hw['vram_gb']}GB", fit_str,
                                 f"{hw['compressed_size_gb']}GB")
        console.print(hw_table)

        # Recommendation
        console.print()
        console.print(Panel(
            f"[bold green]Recommendation:[/] {plan['recommendation']}",
            box=box.ROUNDED,
        ))

        # Per-layer plan summary
        console.print("\n[bold]Per-Layer Compression Plan:[/]")
        lt = Table(box=box.MINIMAL_HEAVY_HEAD, header_style="bold cyan")
        lt.add_column("Layer", justify="right")
        lt.add_column("Mats", justify="right")
        lt.add_column("Params (M)", justify="right")
        lt.add_column("Prunable%", justify="right")
        lt.add_column("SVDable%", justify="right")
        lt.add_column("Action")

        for lidx in sorted(plan["per_layer"].keys()):
            mats = plan["per_layer"][lidx]
            total_params = sum(m["params"] for m in mats)
            prunable = sum(m["savings"] for m in mats if m["action"].startswith("prune"))
            svdable = sum(m["savings"] for m in mats if m["action"].startswith("svd"))

            # Count action types
            actions = [m["action"] for m in mats]
            keep_count = actions.count("keep")
            prune_count = sum(1 for a in actions if "prune" in a)
            svd_count = sum(1 for a in actions if "svd" in a)
            action_str = []
            if prune_count:
                action_str.append(f"[red]prune {prune_count}[/]")
            if svd_count:
                action_str.append(f"[yellow]svd {svd_count}[/]")
            if keep_count:
                action_str.append(f"[green]keep {keep_count}[/]")
            action_display = ", ".join(action_str) if action_str else "[green]all keep[/]"

            lt.add_row(
                str(lidx),
                str(len(mats)),
                f"{total_params / 1e6:.2f}",
                f"{100 * prunable / max(total_params, 1):.0f}",
                f"{100 * svdable / max(total_params, 1):.0f}",
                action_display,
            )
        console.print(lt)

        print()
        if plan["noise_pct"] > 5:
            print(f"  To apply: weightnoise prune {args.model} --keep {max(0.5, 1 - plan['noise_pct']/100):.2f} --method wanda --save")
        else:
            print(f"  Low noise model — no pruning needed for quality preservation.")

    # ── COMMAND: compare ──
    elif args.command == "compare":
        from .inspect import NoiseInspector
        from .visualize import console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        import numpy as np

        console.print(f"\n[bold]Comparing: {args.model_a} vs {args.model_b}[/]")

        a = NoiseInspector(args.model_a, device=args.device,
                           trust_remote_code=args.trust_remote_code).analyze()
        b = NoiseInspector(args.model_b, device=args.device,
                           trust_remote_code=args.trust_remote_code).analyze()

        sa, sb = a["summary"], b["summary"]

        # Summary comparison
        t = Table(box=box.MINIMAL_HEAVY_HEAD, header_style="bold cyan")
        t.add_column("Metric", style="yellow")
        t.add_column(args.model_a, justify="right")
        t.add_column(args.model_b, justify="right")
        t.add_column("Diff", justify="right")

        t.add_row("Total params", f"{sa['total_params_m']:.0f}M", f"{sb['total_params_m']:.0f}M",
                  f"{sb['total_params_m'] - sa['total_params_m']:+.0f}M")
        t.add_row("Noise %", f"{sa['noise_percentage']:.1f}%", f"{sb['noise_percentage']:.1f}%",
                  f"{sb['noise_percentage'] - sa['noise_percentage']:+.1f}%", )
        t.add_row("Noise params", f"{sa['estimated_noise_m']:.0f}M", f"{sb['estimated_noise_m']:.0f}M",
                  f"{sb['estimated_noise_m'] - sa['estimated_noise_m']:+.0f}M")
        t.add_row("Signal params", f"{sa['estimated_signal_m']:.0f}M", f"{sb['estimated_signal_m']:.0f}M",
                  f"{sb['estimated_signal_m'] - sa['estimated_signal_m']:+.0f}M")
        t.add_row("Layers", str(sa['num_layers']), str(sb['num_layers']), "")
        t.add_row("Matrices", str(sa['num_matrices']), str(sb['num_matrices']), "")

        console.print("\n[bold]Overall Comparison:[/]")
        console.print(t)

        # WIT diagnostic: if model_b is a WIT transport of model_a, 
        # anomalously high noise in model_b = bad transfer
        if sb['noise_percentage'] > sa['noise_percentage'] * 2:
            console.print(Panel(
                "[bold red]WIT Diagnostic:[/] Student has significantly more noise than teacher.\n"
                "The transport may have corrupted some layers. Inspect individual layers\n"
                "for anomalously high noise using --layer. These layers need fine-tuning.",
                box=box.ROUNDED,
            ))

        # Per-layer comparison
        console.print("\n[bold]Per-Layer Noise Comparison:[/]")
        lt = Table(box=box.MINIMAL_HEAVY_HEAD, header_style="bold cyan")
        lt.add_column("Layer", justify="right")
        lt.add_column(f"Noise% ({args.model_a[:20]})", justify="right")
        lt.add_column(f"Noise% ({args.model_b[:20]})", justify="right")
        lt.add_column("Diff", justify="right")
        lt.add_column("Verdict")

        a_layers = {k: a[k] for k in a if isinstance(k, int)}
        b_layers = {k: b[k] for k in b if isinstance(k, int)}

        for lidx in sorted(set(list(a_layers.keys()) + list(b_layers.keys()))):
            def avg_noise(layer_results):
                mats = list(layer_results["matrices"].values())
                return np.mean([m["low_importance_pct"] for m in mats]) if mats else 0

            a_pct = avg_noise(a_layers.get(lidx, {"matrices": {}}))
            b_pct = avg_noise(b_layers.get(lidx, {"matrices": {}}))
            diff = b_pct - a_pct

            if diff > 10:
                verdict = "[red]BAD TRANSFER[/]"
            elif diff > 5:
                verdict = "[yellow]WARNING[/]"
            elif diff < -5:
                verdict = "[green]CLEANER[/]"
            else:
                verdict = "[dim]ok[/]"

            lt.add_row(str(lidx) if lidx in a_layers and lidx in b_layers
                       else f"{lidx} ({'A only' if lidx in a_layers else 'B only'})",
                       f"{a_pct:.1f}" if lidx in a_layers else "-",
                       f"{b_pct:.1f}" if lidx in b_layers else "-",
                       f"{diff:+.1f}",
                       verdict)

        console.print(lt)

    # ── COMMAND: prune ──
    elif args.command == "prune":
        from .prune import NoisePruner
        pruner = NoisePruner(args.model, device=args.device,
                              trust_remote_code=args.trust_remote_code)
        result = pruner.prune(method=args.method, keep_ratio=args.keep,
                              threshold=args.threshold, save_path=args.output)
        print(f"\n  Pruning complete:")
        print(f"  Original: {result['original_m']:.0f}M params ({result['original_gb']:.2f} GB)")
        print(f"  Pruned:   {result['pruned_m']:.0f}M params ({result['pruned_gb']:.2f} GB)")
        print(f"  Ratio:    {result['compression_ratio']:.1f}x")
        print(f"  Method:   {result['method']}")
        if args.output:
            print(f"  Saved to: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
