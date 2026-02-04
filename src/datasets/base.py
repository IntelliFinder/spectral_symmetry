"""Point cloud dataset ABC and file parsers."""

from abc import ABC, abstractmethod

import numpy as np


class PointCloudDataset(ABC):
    """Abstract base for point cloud datasets yielding (name, points) tuples."""

    @abstractmethod
    def __iter__(self):
        """Yield (name: str, points: ndarray of shape (N, 3)) tuples."""

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------

def _parse_off(path):
    """Parse an OFF file, return Nx3 vertices."""
    with open(path, 'r') as f:
        header = f.readline().strip()
        if header == 'OFF':
            counts = f.readline().strip().split()
        elif header.startswith('OFF'):
            counts = header[3:].strip().split()
        else:
            raise ValueError(f"Not a valid OFF file: {path}")
        n_verts = int(counts[0])
        verts = []
        for _ in range(n_verts):
            line = f.readline().strip()
            if not line:
                continue
            verts.append([float(x) for x in line.split()[:3]])
    return np.array(verts, dtype=np.float64)


def _parse_ply_ascii(lines, n_verts):
    verts = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
        if len(verts) == n_verts:
            break
    return np.array(verts, dtype=np.float64)


def _parse_ply(path):
    """Parse a PLY file (ASCII or binary_little_endian), return Nx3 vertices."""
    with open(path, 'rb') as f:
        raw = f.read()

    # Parse header
    header_end = raw.index(b'end_header\n') + len(b'end_header\n')
    header_text = raw[:header_end].decode('ascii', errors='replace')
    header_lines = header_text.strip().split('\n')

    fmt = 'ascii'
    n_verts = 0
    for line in header_lines:
        if line.startswith('format'):
            fmt = line.split()[1]
        if line.startswith('element vertex'):
            n_verts = int(line.split()[-1])

    if n_verts == 0:
        return np.zeros((0, 3))

    data = raw[header_end:]
    if fmt == 'ascii':
        text_lines = data.decode('ascii', errors='replace').split('\n')
        return _parse_ply_ascii(text_lines, n_verts)
    elif fmt == 'binary_little_endian':
        # Assume x,y,z are first 3 floats per vertex
        verts = np.frombuffer(data[:n_verts * 12], dtype='<f4').reshape(n_verts, 3)
        return verts.astype(np.float64)
    else:
        raise ValueError(f"Unsupported PLY format: {fmt}")


def _parse_obj(path):
    """Parse an OBJ file, return Nx3 vertices."""
    verts = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float64) if verts else np.zeros((0, 3))


_PARSERS = {
    '.off': _parse_off,
    '.ply': _parse_ply,
    '.obj': _parse_obj,
    '.npy': lambda p: np.load(p).astype(np.float64),
}
