import os
import sys
import traceback
import warnings

# Suppress warnings for cleaner output
warnings.simplefilter("ignore")

# Set environment variables for TPU with debug
print("DEBUG: Setting PJRT_DEVICE to TPU")
os.environ["PJRT_DEVICE"] = "TPU"

print("DEBUG: Setting TPU_VISIBLE_DEVICES to 0,1,2,3 (based on ls output)")
os.environ["TPU_VISIBLE_DEVICES"] = "0,1,2,3"  # Adjusted to match your ls output (accel0-3)

# Additional XLA flags for debugging and potential fix
print("DEBUG: Setting XLA_FLAGS for core dump and debug")
os.environ["XLA_FLAGS"] = (
    "--xla_tpu_debug_allow_core_dump=true "
    "--xla_tpu_debug_dump_anomalies_only=true "
    "--xla_force_host_platform_device_count=1"  # Fallback to simulate single device if needed
)

# Try to import torch_xla modules with version check
try:
    import torch_xla
    print(f"DEBUG: torch_xla version: {torch_xla.__version__}")
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    from torch_xla.distributed.parallel_loader import MpDeviceLoader
    print("DEBUG: Successfully imported torch_xla modules")
except ImportError as e:
    print("ERROR: Failed to import torch_xla modules")
    print(f"ERROR DETAILS: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

import torch
print(f"DEBUG: torch version: {torch.__version__}")

def _test_fn(rank: int):
    # Log rank and process info
    print(f"DEBUG [Rank {rank}]: Starting test function")
    print(f"DEBUG [Rank {rank}]: Process ID: {os.getpid()}")
    print(f"DEBUG [Rank {rank}]: XLA ordinal: {xm.get_ordinal()}")
    print(f"DEBUG [Rank {rank}]: XLA world size: {xm.xrt_world_size()}")

    # List all /dev/accel* files for debug
    try:
        accel_files = [f for f in os.listdir('/dev') if f.startswith('accel')]
        print(f"DEBUG [Rank {rank}]: Available accel devices: {accel_files}")
    except Exception as e:
        print(f"ERROR [Rank {rank}]: Failed to list /dev/accel*")
        print(f"ERROR DETAILS: {str(e)}")

    # Get supported devices with error handling
    try:
        devices = xm.get_xla_supported_devices("TPU")
        print(f"DEBUG [Rank {rank}]: Supported TPU devices: {devices}")
        if not devices:
            raise ValueError("No TPU devices found")
    except Exception as e:
        print(f"ERROR [Rank {rank}]: Failed to get supported devices")
        print(f"ERROR DETAILS: {str(e)}")
        traceback.print_exc()
        return

    # Get XLA device
    try:
        device = xm.xla_device()
        print(f"DEBUG [Rank {rank}]: XLA device: {device}")
    except Exception as e:
        print(f"ERROR [Rank {rank}]: Failed to get XLA device")
        print(f"ERROR DETAILS: {str(e)}")
        traceback.print_exc()
        return

    # Simple tensor operation on TPU
    try:
        # Create tensors on XLA device
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        print(f"DEBUG [Rank {rank}]: Created tensors on {device}")

        # Perform matrix multiplication
        c = torch.matmul(a, b)
        print(f"DEBUG [Rank {rank}]: Matrix multiplication completed. Result shape: {c.shape}")

        # Simple model and optimizer test
        model = torch.nn.Linear(1024, 1024).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        loss = c.sum()
        loss.backward()
        xm.optimizer_step(optimizer)
        print(f"DEBUG [Rank {rank}]: Optimizer step completed successfully")
    except Exception as e:
        print(f"ERROR [Rank {rank}]: Failed during tensor operations")
        print(f"ERROR DETAILS: {str(e)}")
        traceback.print_exc()
        return

    # Barrier to sync all ranks
    try:
        xm.rendezvous("test_completion")
        print(f"DEBUG [Rank {rank}]: Rendezvous completed - all ranks synced")
    except Exception as e:
        print(f"ERROR [Rank {rank}]: Rendezvous failed")
        print(f"ERROR DETAILS: {str(e)}")
        traceback.print_exc()

    print(f"DEBUG [Rank {rank}]: Test completed successfully")

if __name__ == "__main__":
    print("DEBUG: Starting main script")
    try:
        # Spawn on all available devices (nprocs=None)
        xmp.spawn(_test_fn, nprocs=None)
        print("DEBUG: Spawn completed successfully")
    except Exception as e:
        print("ERROR: Failed to spawn processes - Falling back to single process test")
        print(f"ERROR DETAILS: {str(e)}")
        traceback.print_exc()

        # Fallback: Run test in single process mode (rank 0 only)
        print("DEBUG: Starting single process fallback test")
        _test_fn(0)
        print("DEBUG: Single process test completed")