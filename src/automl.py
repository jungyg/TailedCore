import psutil
import GPUtil


def get_memory_size(device, memory_type="free", device_index=None):

    if device == "cpu":
        memories = psutil.virtual_memory()
        return getattr(memories, memory_type)
    elif device == "gpu":
        assert memory_type in ["free", "available"]
        free_gpu_memory = GPUtil.getGPUs()[device_index].memoryFree * 1024**2
        return free_gpu_memory
    else:
        raise ValueError("Invalid device type")


def get_max_len_feas(
    fea_dim, device="cpu", max_usage=0.25, memory_type="free", device_index=None
):
    usage_one_fea = fea_dim * 4

    memory = get_memory_size("cpu", memory_type=memory_type, device_index=device_index)
    max_used_memory = memory * max_usage

    if device == "cpu":
        memories = psutil.virtual_memory()
        memory = getattr(memories, memory_type)
        max_used_memory = memory * max_usage
    elif device == "gpu":
        pass

    max_len_feas = int(max_used_memory / usage_one_fea)

    return max_len_feas


def get_max_coreset_ratio(fea_dim, len_feas):
    max_len_feas = get_max_len_feas(fea_dim)
    return min(max_len_feas / len_feas, 0.1)


def calculate_memory_usage(matrix):
    # Get the number of elements in the matrix
    num_elements = matrix.size

    # Get the size of each element in bytes
    element_size = matrix.itemsize

    # Calculate total memory usage in bytes
    memory_usage_bytes = num_elements * element_size

    # Convert to kilobytes (optional)

    return memory_usage_bytes
