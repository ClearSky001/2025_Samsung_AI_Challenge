import wandb
import yaml
import argparse
import subprocess
import os

def run_sweep():
    """
    Initialize and run wandb sweep for hyperparameter tuning
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_config", default="sweep_config.yaml", help="Path to sweep configuration file")
    parser.add_argument("--count", type=int, default=20, help="Number of runs to execute")
    parser.add_argument("--project", default="blip-vqav2-finetuning", help="Wandb project name")
    args = parser.parse_args()
    
    # Load sweep configuration
    with open(args.sweep_config, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Update project name
    if args.project:
        if 'fixed_parameters' not in sweep_config:
            sweep_config['fixed_parameters'] = {}
        sweep_config['fixed_parameters']['wandb_project'] = args.project
    
    # Initialize sweep
    print(f"Initializing sweep with config: {args.sweep_config}")
    print(f"Project: {args.project}")
    print(f"Number of runs: {args.count}")
    
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    print(f"Sweep ID: {sweep_id}")
    print(f"Sweep URL: https://wandb.ai/{wandb.api.default_entity}/{args.project}/sweeps/{sweep_id}")
    
    # Start sweep agent
    print("Starting sweep agent...")
    wandb.agent(sweep_id, count=args.count)
    
    print("Sweep completed!")

def run_single_experiment():
    """
    Run a single experiment with default parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="blip-vqav2-finetuning", help="Wandb project name")
    parser.add_argument("--name", default="single-experiment", help="Experiment name")
    parser.add_argument("--max_train_samples", type=int, default=1000, help="Number of training samples for quick test")
    parser.add_argument("--max_val_samples", type=int, default=500, help="Number of validation samples for quick test")
    args = parser.parse_args()
    
    # Run single experiment
    cmd = [
        "python", "blip_finetune.py",
        "--wandb_project", args.project,
        "--wandb_name", args.name,
        "--max_train_samples", str(args.max_train_samples),
        "--max_val_samples", str(args.max_val_samples),
        "--num_train_epochs", "2",
        "--per_device_train_batch_size", "8",
        "--per_device_eval_batch_size", "8",
        "--learning_rate", "2e-5",
        "--logging_steps", "10"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        # Remove 'single' from args before parsing
        sys.argv.pop(1)
        run_single_experiment()
    else:
        run_sweep() 