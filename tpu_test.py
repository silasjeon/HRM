import os
import sys
import traceback

# Set environment variables for TPU
os.environ["PJRT_DEVICE"] = "TPU"
print("DEBUG: PJRT_DEVICE set to TPU")

# Try to import torch_xla modules with debug logging
try:
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
print("DEBUG: Imported torch")

def _test_fn(rank: int):
    # Log rank and process info
    print(f"DEBUG [Rank {rank}]: Starting test function")
    print(f"DEBUG [Rank {rank}]: XLA ordinal: {xm.get_ordinal()}")
    print(f"DEBUG [Rank {rank}]: XLA world size: {xm.xrt_world_size()}")

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
        print("ERROR: Failed to spawn processes")
        print(f"ERROR DETAILS: {str(e)}")
        traceback.print_exc()
