def has_batch_dim(axes: str) -> bool:
    try:
        index = axes.index("b")
    except ValueError:
        return False
    else:
        if index != 0:
            raise ValueError("Batch dimension is only supported in first position")
        return True
