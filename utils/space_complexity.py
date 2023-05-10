


def get_model_size_in_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size = (param_size + buffer_size) / 1024**2
    
    return size


def get_tensor_size_in_mb(tensor_):
    
    tensor_size = tensor_.element_size()*tensor_.nelement()
    
    tensor_size = tensor_size / 1024**2
    
    return tensor_size