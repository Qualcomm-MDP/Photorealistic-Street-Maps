import numpy as np

def clip_confidence_score(confidence_raw):
    confidence_8bit = np.clip(
        (confidence_raw * 255),
        0,
        255
    ).astype(np.uint8)
    return confidence_8bit

def confidence_score(distance, cos, clear, average=True):
    # Going to need to figure out a better metric, CHAT suggested an exponential decay function
    if average:
        distance = np.mean(distance)
    distance_confidence = 1.0 / (1.0 + distance / 15.0)
    confidence = cos * clear * distance_confidence
    return clip_confidence_score(confidence)