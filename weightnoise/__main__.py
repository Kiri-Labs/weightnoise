#!/usr/bin/env python3
"""CLI entry point for weightnoise."""


def main():
    import argparse, sys

    parser = argparse.ArgumentParser(
        prog="weightnoise",
        description="Visualize and remove noise from neural network weights."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Inspect
    inspect_p = sub.add_parser("inspect", help="Inspect model weights for noise")
    inspect_p.add_argument("model", help="HF model ID or local path")
    inspect_p.add_argument("--layer", type=int, default=None, help="Single layer to inspect")
    inspect_p.add_argument("--threshold", type=float, default=0.01,
                           help="Noise threshold (fraction of max singular value)")
    inspect_p.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    inspect_p.add_argument("--trust-remote-code", action="store_true",
                           help="Trust remote code for custom architectures")

    # Prune
    prune_p = sub.add_parser("prune", help="Remove noise from model weights")
    prune_p.add_argument("model", help="HF model ID or local path")
    prune_p.add_argument("--output", "-o", default=None, help="Output path for pruned model")
    prune_p.add_argument("--keep", type=float, default=0.5,
                         help="Fraction of weights to keep (1.0 = none removed)")
    prune_p.add_argument("--method", default="magnitude",
                         choices=["magnitude", "spectral", "wanda"],
                         help="Pruning method")
    prune_p.add_argument("--threshold", type=float, default=0.01,
                         help="Noise threshold for spectral method")
    prune_p.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    prune_p.add_argument("--trust-remote-code", action="store_true",
                         help="Trust remote code for custom architectures")
    prune_p.add_argument("--save", action="store_true",
                         help="Save pruned model to disk")

    args = parser.parse_args()

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

        total = results["summary"]["total_params_m"]
        noise_pct = results["summary"]["noise_percentage"]
        print(f"\n  Total parameters: {total:.0f}M")
        print(f"  Estimated noise:  {noise_pct:.1f}%")
        print(f"  Could compress to ~{total * (1 - noise_pct/100):.0f}M params")

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
        if args.output:
            print(f"  Saved to: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
