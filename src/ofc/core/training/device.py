"""GPU detection and device management utilities."""

from typing import Optional

try:
    import torch
except ImportError:
    torch = None


def detect_gpu() -> Optional[dict]:
    """
    Detect available GPU and return information.
    
    Returns:
        Dictionary with GPU info: {"name": str, "memory_gb": float, "device": str}
        or None if no GPU available
    """
    if torch is None:
        return None
    
    try:
        if torch.cuda.is_available():
            device_id = 0  # Default to first GPU
            name = torch.cuda.get_device_name(device_id)
            
            # Get memory info if available
            memory_gb = None
            try:
                memory_bytes = torch.cuda.get_device_properties(device_id).total_memory
                memory_gb = memory_bytes / (1024 ** 3)  # Convert to GB
            except Exception:
                pass
            
            return {
                "name": name,
                "memory_gb": memory_gb,
                "device": f"cuda:{device_id}",
                "device_id": device_id,
            }
        
        # Check for Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return {
                "name": "Apple Silicon (MPS)",
                "memory_gb": None,  # MPS doesn't expose memory info easily
                "device": "mps",
                "device_id": 0,
            }
    except Exception:
        pass
    
    return None


def is_gpu_available() -> bool:
    """
    Check if GPU is available.
    
    Returns:
        True if GPU is available, False otherwise
    """
    if torch is None:
        return False
    
    try:
        return torch.cuda.is_available() or (
            hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        )
    except Exception:
        return False


def get_device(use_gpu: bool, device_string: Optional[str] = None) -> "torch.device":
    """
    Get the appropriate torch.device based on preferences.
    
    Args:
        use_gpu: Whether to use GPU if available
        device_string: Optional specific device string (e.g., "cuda:0", "mps", "cpu")
    
    Returns:
        torch.device instance
    """
    if torch is None:
        raise ImportError("PyTorch is not installed")
    
    # If specific device string provided, use it
    if device_string:
        try:
            return torch.device(device_string)
        except Exception:
            # Fallback to CPU if device string is invalid
            return torch.device("cpu")
    
    # Otherwise, determine based on use_gpu flag
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            # GPU requested but not available, fallback to CPU
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def list_available_devices() -> list[dict]:
    """
    List all available devices (CPU, CUDA GPUs, MPS).
    
    Returns:
        List of device dictionaries, each with:
        - "name": Display name
        - "device": Device string (e.g., "cpu", "cuda:0", "mps")
        - "memory_gb": Memory in GB (if available)
    """
    devices = []
    
    # Always add CPU
    devices.append({
        "name": "CPU",
        "device": "cpu",
        "memory_gb": None,
    })
    
    if torch is None:
        return devices
    
    try:
        # Add CUDA devices
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                try:
                    name = torch.cuda.get_device_name(i)
                    memory_gb = None
                    try:
                        memory_bytes = torch.cuda.get_device_properties(i).total_memory
                        memory_gb = memory_bytes / (1024 ** 3)
                    except Exception:
                        pass
                    
                    devices.append({
                        "name": name,
                        "device": f"cuda:{i}",
                        "memory_gb": memory_gb,
                    })
                except Exception:
                    # Skip devices we can't get info for
                    devices.append({
                        "name": f"CUDA:{i}",
                        "device": f"cuda:{i}",
                        "memory_gb": None,
                    })
        
        # Add MPS if available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append({
                "name": "Apple Silicon (MPS)",
                "device": "mps",
                "memory_gb": None,
            })
    except Exception:
        pass
    
    return devices
