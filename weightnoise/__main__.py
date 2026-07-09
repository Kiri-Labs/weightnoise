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
    inspect_p.add_argument("--threshold", type=float, default=None,
                           help="Noise threshold override. Default: adaptive (5th percentile of scores)")
    inspect_p.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    inspect_p.add_argument("--trust-remote-code", action="store_true",
                           help="Trust remote code for custom architectures")

    # ── plan ──
    plan_p = sub.add_parser("plan", help="Generate compression plan from noise analysis")
    plan_p.add_argument("model", help="HF model ID or local path")
    plan_p.add_argument("--vram", type=float, default=None,
                        help="Target VRAM in GB. Default: auto-detect from hardware")
    plan_p.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8", "int4"],
                        help="Target inference precision (default fp16)")
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
                         choices=None,
                         help="Pruning method. Built-in: magnitude, spectral, wanda. "
                              "Pass any string -- method plugins in development.")
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
        print(f"  Threshold: {summary.get('analysis_threshold', 'adaptive')}")
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
        results = inspector.analyze()  # auto threshold
        plan = generate_plan(results, vram_gb=args.vram, precision=args.precision)

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

        # Hardware targets (dynamically detected — no hardcoded list)
        console.print("\n[bold]Hardware Targets:[/]")
        hw_table = Table(box=box.MINIMAL_HEAVY_HEAD, header_style="bold cyan")
        hw_table.add_column("Target")
        hw_table.add_column("VRAM", justify="right")
        hw_table.add_column("Fits?", justify="center")
        hw_table.add_column("Compressed Size", justify="right")

        for key, hw in plan["hardware_targets"].items():
            if hw["fits"]:
                fit_str = "[green]YES[/]"
            elif hw.get("needs_quant"):
                fit_str = f"[yellow]as {hw['needs_quant']}[/]"
            else:
                fit_str = "[red]NO[/]"
            hw_table.add_row(key, f"{hw['vram_gb']}GB", fit_str,
                             f"{hw['compressed_size_gb']}GB")
        console.print(hw_table)
        if args.vram:
            console.print(f"  (--vram {args.vram}GB @ {args.precision})")

        # Threshold note
        thresh = plan.get("thresholds", {})
        console.print()
        console.print(f"[dim]Thresholds: {thresh.get('prune_at', 'adaptive')} | "
                      f"{thresh.get('svd_at', 'adaptive')}[/]")

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
        savings = plan.get("savings", {})
        total_savings = savings.get("total_m", 0)
        if total_savings > 1:
            keep_ratio = round(max(0.5, 1 - total_savings / max(plan["total_m"], 1) * 2), 2)
            print(f"  To apply: weightnoise prune {args.model} --keep {keep_ratio} --method wanda --save")
        else:
            print(f"  Low noise model \u2014 minimal pruning needed for quality preservation.")

    # ── COMMAND: compare ──
    elif args.command == "compare":
        from .inspect import NoiseInspector
        from .visualize import console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        import numpy as np
        model_a, model_b = args.model_a, args.model_b

        console.print(f"\n[bold]Comparing: {model_a} vs {model_b}[/]")

        a = NoiseInspector(model_a, device=args.device,
                           trust_remote_code=args.trust_remote_code).analyze()
        b = NoiseInspector(model_b, device=args.device,
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

        # WIT diagnostic: compute noise difference dynamically
        noise_diffs = []
        for lidx in set(list(a_layers.keys()) + list(b_layers.keys())):
            def avg_noise(lr):
                mats = list(lr.get("matrices", {}).values())
                return np.mean([m["low_importance_pct"] for m in mats]) if mats else 0
            a_pct = avg_noise(a_layers.get(lidx, {}))
            b_pct = avg_noise(b_layers.get(lidx, {}))
            if a_pct > 0:
                noise_diffs.append(b_pct - a_pct)
        
        if noise_diffs:
            median_diff = np.median(noise_diffs)
            mad = np.median(np.abs(noise_diffs - median_diff))
            anomaly_threshold = median_diff + 2 * mad if mad > 0 else median_diff + 10
            worst_diff = max(noise_diffs) if noise_diffs else 0
            
            if worst_diff > anomaly_threshold:
                n_anomalies = sum(1 for d in noise_diffs if d > anomaly_threshold)
                console.print(Panel(
                    f"[bold yellow]WIT Diagnostic:[/] {n_anomalies} layer(s) show anomalous noise increase\n"
                    f"(>{anomaly_threshold:.1f}% above median diff). Inspect with --layer for fine-tuning targets.",
                    box=box.ROUNDED,
                ))

        # Per-layer comparison
        console.print("\n[bold]Per-Layer Noise Comparison:[/]")
        lt = Table(box=box.MINIMAL_HEAVY_HEAD, header_style="bold cyan")
        lt.add_column("Layer", justify="right")
        lt.add_column(f"Noise% ({model_a[:20]})", justify="right")
        lt.add_column(f"Noise% ({model_b[:20]})", justify="right")
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

            # Adaptive verdict: use MAD from the distribution
            if noise_diffs:
                if diff > anomaly_threshold:
                    verdict = "[red]BAD TRANSFER[/]"
                elif diff > median_diff + mad:
                    verdict = "[yellow]WARNING[/]"
                elif diff < median_diff - mad:
                    verdict = "[green]CLEANER[/]"
                else:
                    verdict = "[dim]ok[/]"
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
    # compress
    compress_p = sub.add_parser("compress", help="Compress teacher model into student via WIT spectral stitching")
    compress_p.add_argument("teacher", help="Large teacher model")
    compress_p.add_argument("student", help="Small student architecture")
    compress_p.add_argument("--save", "-o", default=None, help="Save compressed model path")
    compress_p.add_argument("--stream", action="store_true",
                            help="Stream teacher weights shard-by-shard (avoids OOM for 100B+ models)")
    compress_p.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    compress_p.add_argument("--trust-remote-code", action="store_true",
                            help="Trust remote code for custom architectures")


    # compress
    elif args.command == "compress":
        from .stitch import compress as stitch_compress
        
        print(f"\n  Compressing teacher -> student")
        print(f"  Teacher: {args.teacher}")
        print(f"  Student: {args.student}")
        result = stitch_compress(
            args.teacher, args.student,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            save_path=args.save,
            streaming=args.stream,
        )
        
        print(f"\n  Compression result:")
        print(f"  Student params: {result['student_params_m']:.0f}M")
        print(f"  Ratio: {result['compression_ratio']:.1f}x")
        print(f"  Matrices stitched: {result['matrices_stitched']}")
        if args.save:
            print(f"  Saved to: {args.save}")
