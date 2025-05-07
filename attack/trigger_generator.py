def add_cross_trigger(images, labels, num=6, trigger_size=4, intensity=0.5):
    """
        Add a cross-shaped trigger in the top-left corner of the images
    """
    if trigger_size > 0:
        mid = trigger_size // 2
        images[:num, :, mid:mid+1, :trigger_size] = intensity
        images[:num, :, :trigger_size, mid:mid+1] = intensity
        labels[:num] = 0

    return images, labels