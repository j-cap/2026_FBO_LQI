"""Version-2 reference assignment; the archived v1 diagnostic is immutable."""
import numpy as np


def stable_minimum(values, labels, *, tolerance=1e-12):
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Need a nonempty finite set of distances.')
    best = float(values.min())
    tied = np.flatnonzero(values <= best+tolerance*max(1., abs(best)))
    return int(min(tied, key=lambda i: labels[i]))


def nearest_reference(metadata, controlled_metadata, *, tolerance=1e-12):
    """Use archived reference-only descriptors, with deterministic near-ties."""
    centroid = max(metadata['spectral_centroid_hz'], 1e-6)
    distances = [((metadata['speed']-r['speed'])/10.)**2+
                 np.log(centroid/max(r['spectral_centroid_hz'], 1e-6))**2
                 for r in controlled_metadata]
    labels = [r['id'] for r in controlled_metadata]
    return labels[stable_minimum(distances, labels, tolerance=tolerance)]
