#!/usr/bin/env python3
import sys
import os
import numpy as np
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
from shapely import wkt
import time
from datetime import datetime
import json
import signal
import sys

# Add the parent directory to the Python path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.utils import load_data, validate_polygon_geometry, save_data

def print_progress_header():
    """Print a formatted progress header"""
    print("\n" + "="*80)
    print(f"{'Simulation Progress Monitor':^80}")
    print("="*80)
    print(f"{'Time':<12} {'Completed':<10} {'Running':<8} {'Failed':<8} {'Progress':<10} {'ETA':<12}")
    print("-"*80)

def print_progress_line(progress):
    """Print a single progress line"""
    eta_str = f"{progress['eta']/60:.1f}m" if progress['eta'] > 0 else "N/A"
    progress_str = f"{progress['completed_pct']:.1f}%"
    time_str = datetime.now().strftime("%H:%M:%S")
    
    print(f"{time_str:<12} {progress['completed']:<10} {progress['running']:<8} "
          f"{progress['failed']:<8} {progress_str:<10} {eta_str:<12}")

def run_single_simulation(args):
    """Run a single simulation with specified number of cores"""
    data_dir_name, data_parent_dir, cores_per_sim = args
    
    print(f"Processing {data_dir_name} with {cores_per_sim} cores")
    
    data_path = os.path.join(data_parent_dir, data_dir_name)
    start_time = time.time()
    
    try:
        # Load and validate data
        toolpath, _, metadata, inside_outside = load_data(data_path, output_name="metadata_00")
        polygon = wkt.loads(metadata['shape'])
        
        # Skip if max_T.npy already exists
        if os.path.isfile(os.path.join(data_path, "max_T.npy")):
            return {
                "sim_name": data_dir_name,
                "status": "skipped",
                "message": "Already processed",
                "start_time": start_time,
                "end_time": time.time()
            }
        
        # Create a separate Python script call for this simulation
        script_path = os.path.join(os.path.dirname(__file__), "run_single_sim.py")
        
        # Run the simulation as a subprocess with specific number of MPI ranks
        cmd = [
            "mpirun", "-n", str(cores_per_sim),
            "python", script_path,
            "--data_path", data_path
        ]
        
        # Run with real-time output streaming
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time with simulation name prefix
        output_lines = []
        try:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    if line:  # Only print non-empty lines
                        print(f"[{data_dir_name}] {line}")
                        output_lines.append(line)
            
            # Wait for process to complete
            process.wait(timeout=3600)  # 1 hour timeout
            
            if process.returncode == 0:
                return {
                    "sim_name": data_dir_name,
                    "status": "completed",
                    "message": "Success",
                    "start_time": start_time,
                    "end_time": time.time(),
                    "output_lines": len(output_lines)
                }
            else:
                return {
                    "sim_name": data_dir_name,
                    "status": "failed",
                    "message": f"Return code: {process.returncode}",
                    "start_time": start_time,
                    "end_time": time.time(),
                    "output_lines": len(output_lines)
                }
                
        except subprocess.TimeoutExpired:
            process.kill()
            return {
                "sim_name": data_dir_name,
                "status": "failed",
                "message": "Timeout (1 hour)",
                "start_time": start_time,
                "end_time": time.time()
            }
            
    except Exception as e:
        return {
            "sim_name": data_dir_name,
            "status": "failed",
            "message": str(e),
            "start_time": start_time,
            "end_time": time.time()
        }

def signal_handler(sig, frame):
    print(f"\n\n⚠️  RECEIVED INTERRUPT SIGNAL - GRACEFULLY SHUTTING DOWN...")
    print("This may take a few seconds to clean up processes...")
    sys.exit(0)

def main():
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description='Run thermal simulations in parallel')
    parser.add_argument('--data_dir', type=str, default="/mnt/d/sim_data/DATABASE_OPT",
                       help='Parent directory containing simulation data directories')
    parser.add_argument('--max_parallel', type=int, default=None,
                       help='Maximum number of simulations to run in parallel (default: number of CPU cores)')
    parser.add_argument('--cores_per_sim', type=int, default=1,
                       help='Number of MPI cores per simulation (default: 1)')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Log file to save detailed progress (optional)')
    parser.add_argument('--monitor_interval', type=int, default=10,
                       help='Progress update interval in seconds (default: 10)')
    args = parser.parse_args()
    
    data_parent_dir = args.data_dir
    max_parallel = args.max_parallel or mp.cpu_count()
    cores_per_sim = args.cores_per_sim
    
    print(f"Data directory: {data_parent_dir}")
    print(f"Max parallel simulations: {max_parallel}")
    print(f"Cores per simulation: {cores_per_sim}")
    
    # Find all directories to process
    all_dirs = [d for d in os.listdir(data_parent_dir) 
                if os.path.isdir(os.path.join(data_parent_dir, d))]
    all_dirs.sort()
    
    # Filter out already processed directories
    filtered_dirs = []
    for d in all_dirs:
        dir_path = os.path.join(data_parent_dir, d)
        if not os.path.isfile(os.path.join(dir_path, "max_T.npy")):
            filtered_dirs.append(d)
    
    print(f"Found {len(filtered_dirs)} directories to process:")
    for i, dir_name in enumerate(filtered_dirs):
        print(f"  {i+1:3d}: {dir_name}")
    
    if not filtered_dirs:
        print("No directories to process!")
        return
    
    # Ask for confirmation
    response = input(f"\nProcess {len(filtered_dirs)} directories with {max_parallel} parallel simulations? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Prepare arguments for parallel execution
    sim_args = [(dir_name, data_parent_dir, cores_per_sim) for dir_name in filtered_dirs]
    
    # Run simulations in parallel
    print(f"\n{'='*80}")
    print(f"STARTING {len(filtered_dirs)} SIMULATIONS IN PARALLEL")
    print(f"Max parallel: {max_parallel} | Cores per simulation: {cores_per_sim}")
    print(f"{'='*80}")
    print("Real-time output from individual simulations will be shown with [SIM_NAME] prefix")
    print("Progress summaries will be shown when each simulation completes")
    print(f"{'='*80}\n")
    
    results = []
    completed = 0
    failed = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all tasks
        future_to_sim = {
            executor.submit(run_single_simulation, args): args[0] 
            for args in sim_args
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_sim):
            sim_name = future_to_sim[future]
            try:
                result = future.result()
                results.append(result)
                
                # Update counters based on result
                if result['status'] == 'completed':
                    completed += 1
                    print(f"\n✅ COMPLETED: {result['sim_name']} ({(result['end_time']-result['start_time'])/60:.1f}m, {result.get('output_lines', 0)} lines)")
                elif result['status'] == 'skipped':
                    completed += 1  # Count skipped as completed
                    print(f"\n⏭️  SKIPPED: {result['sim_name']} - {result['message']}")
                elif result['status'] == 'failed':
                    failed += 1
                    print(f"\n❌ FAILED: {result['sim_name']} - {result['message']}")
                
                # Print progress summary line
                elapsed = time.time() - start_time
                total_done = completed + failed
                progress_pct = (total_done / len(filtered_dirs)) * 100 if len(filtered_dirs) > 0 else 0
                eta = (elapsed / max(total_done, 1)) * (len(filtered_dirs) - total_done) if total_done > 0 else 0
                eta_str = f"{eta/60:.1f}m" if eta > 0 else "N/A"
                
                time_str = datetime.now().strftime("%H:%M:%S")
                print(f"PROGRESS: {time_str} | {completed}/{len(filtered_dirs)} completed | {failed} failed | {progress_pct:.1f}% | ETA: {eta_str}")
                print("=" * 80)
                
                # Save log if requested
                if args.log_file:
                    log_data = {
                        'timestamp': datetime.now().isoformat(),
                        'progress': {
                            'completed': completed,
                            'failed': failed,
                            'running': len(filtered_dirs) - total_done,
                            'total': len(filtered_dirs),
                            'completed_pct': progress_pct,
                            'elapsed': elapsed,
                            'eta': eta
                        },
                        'results': results
                    }
                    with open(args.log_file, 'w') as f:
                        json.dump(log_data, f, indent=2)
                        
            except Exception as e:
                failed += 1
                error_result = {
                    "sim_name": sim_name,
                    "status": "failed",
                    "message": f"Exception: {str(e)}",
                    "start_time": time.time(),
                    "end_time": time.time()
                }
                results.append(error_result)
                print(f"❌ {sim_name:<30} Exception - {str(e)}")
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    # Group results by status
    completed_sims = [r for r in results if r['status'] in ['completed', 'skipped']]
    failed_sims = [r for r in results if r['status'] == 'failed']
    
    print(f"✅ Completed ({len(completed_sims)}):")
    for result in completed_sims:
        runtime = (result['end_time'] - result['start_time']) / 60
        print(f"   {result['sim_name']} ({runtime:.1f}m) - {result['message']}")
    
    if failed_sims:
        print(f"\n❌ Failed ({len(failed_sims)}):")
        for result in failed_sims:
            runtime = (result['end_time'] - result['start_time']) / 60
            print(f"   {result['sim_name']} ({runtime:.1f}m) - {result['message']}")
    
    # Summary statistics
    total_time = time.time() - start_time
    print(f"\nSummary:")
    print(f"  Total simulations: {len(filtered_dirs)}")
    print(f"  Completed: {len(completed_sims)}")
    print(f"  Failed: {len(failed_sims)}")
    print(f"  Success rate: {(len(completed_sims)/len(filtered_dirs)*100):.1f}%")
    print(f"  Total time: {total_time/60:.1f} minutes")
    
    if args.log_file:
        print(f"  Detailed log saved to: {args.log_file}")

if __name__ == "__main__":
    main()