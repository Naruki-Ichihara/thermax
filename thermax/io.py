"""Incremental HDF5 snapshots of a running optimisation.

One file per run, written as it goes rather than at the end, so an interrupted
run still leaves everything up to the last snapshot readable. The file carries
the run's parameters as root attributes, which is what makes a parameter sweep
tractable: each file says what produced it, and nothing has to be recovered
from the directory name.

Layout::

    /                       attrs: every scalar/string/sequence passed as
                                   ``attrs``, plus n_nodes, n_cells, created
    /mesh/points            (n_nodes, dim)      float32, written once
    /mesh/cells             (n_cells, nodes)    int32,   written once
    /fields/<name>          (N, n_nodes)        float32, one row per snapshot
    /scalars/<name>         (N,)                float64, one value per snapshot
    /iteration              (N,)                int32

``N`` grows with :meth:`SnapshotWriter.append`; every dataset is chunked and
resizable along axis 0, and the file is flushed after each append.
"""

from __future__ import annotations

import datetime
import os
from typing import Dict, Sequence

import numpy as onp

__all__ = ["SnapshotWriter", "snapshots_to_xdmf"]


class SnapshotWriter:
    """Append-only HDF5 writer for design-field snapshots.

    Parameters
    ----------
    path : str
        Output file. Overwritten if it exists.
    mesh : object with ``points`` and ``cells``
        Written once, so a reader needs nothing but this file to plot.
    field_names : sequence of str
        Names for the per-node fields handed to :meth:`append`, in order. Each
        may be ``(n_nodes,)`` or ``(n_nodes, k)``; the dataset is created from
        the shape of the first append.
    attrs : dict
        Run parameters. Values that are not numbers, strings or arrays of
        those are stored as their ``repr`` rather than dropped -- a sweep
        should not silently lose the one setting that was varied.
    compression : str or None
        Passed to h5py. ``"gzip"`` roughly halves these files and costs little,
        since the writes are small and infrequent.
    xdmf : bool
        Also drive a :class:`feax.XDMFWriter` alongside, writing
        ``<stem>_evolution.xdmf`` + ``.h5`` next to this file. That pair is what
        ParaView opens: plain HDF5 is not a format it reads, and the XDMF XML is
        the index that tells it which dataset is the mesh, which are the fields,
        and which frame is which time step.

        Two files rather than one because they answer different questions. This
        one stores every field as ``(N, n_nodes)`` with the run's parameters as
        attributes, which is what a sweep wants to slice in numpy; the XDMF pair
        stores one group per frame, which is what the XDMF format requires. The
        duplication is a few MB.
    """

    def __init__(self, path: str, mesh, field_names: Sequence[str],
                 attrs: Dict = None, compression: str = "gzip",
                 xdmf: bool = True):
        import h5py                      # imported here: only snapshotting needs it

        self._f = h5py.File(path, "w")
        self._names = tuple(field_names)
        self._n = 0
        self._last_iteration = None
        self.path = path

        pts = onp.asarray(mesh.points)
        cells = onp.asarray(mesh.cells)
        g = self._f.create_group("mesh")
        g.create_dataset("points", data=pts.astype(onp.float32),
                         compression=compression)
        g.create_dataset("cells", data=cells.astype(onp.int32),
                         compression=compression)

        self._f.attrs["n_nodes"] = pts.shape[0]
        self._f.attrs["n_cells"] = cells.shape[0]
        self._f.attrs["created"] = datetime.datetime.now().isoformat(
            timespec="seconds")
        for k, v in (attrs or {}).items():
            self._f.attrs[k] = _as_attr(v)

        # Field datasets are created on the FIRST append, from the shapes that
        # actually arrive, so a vector field (a director, a displacement) is
        # stored as (N, n_nodes, k) rather than being flattened into a row.
        self._n_nodes = pts.shape[0]
        self._compression = compression
        self._fields = self._f.create_group("fields")
        self._scalars = self._f.create_group("scalars")
        self._f.create_dataset("iteration", shape=(0,), maxshape=(None,),
                               dtype="i4", chunks=(64,))
        self._f.flush()

        self._xdmf = None
        if xdmf:
            from feax.utils import XDMFWriter
            stem = os.path.splitext(path)[0]
            # NOT "<stem>.xdmf": XDMFWriter derives its companion HDF5 name by
            # swapping the extension, which would collide with this file.
            self._xdmf = XDMFWriter(mesh, stem + "_evolution.xdmf")
            self._xdmf.__enter__()

    def append(self, iteration: int, fields: Sequence, scalars: Dict = None):
        """Add one snapshot. ``fields`` must match ``field_names`` in order."""
        if len(fields) != len(self._names):
            raise ValueError(
                f"SnapshotWriter.append: expected {len(self._names)} fields "
                f"{self._names}, got {len(fields)}")
        # A repeated iteration index is a no-op, not an error. The caller writes
        # one final snapshot after the run, and when the run happens to end on a
        # multiple of the snapshot period that call duplicates the periodic one.
        # XDMF requires strictly increasing times, and a duplicate makes the
        # reader drop a time value while the frame stays in the file -- the last
        # frame then reads back empty.
        if self._last_iteration is not None and iteration <= self._last_iteration:
            return
        self._last_iteration = int(iteration)
        i = self._n
        for name, arr in zip(self._names, fields):
            arr = onp.asarray(arr, dtype=onp.float32)
            if arr.shape[0] != self._n_nodes:
                raise ValueError(
                    f"SnapshotWriter.append: field {name!r} has leading "
                    f"dimension {arr.shape[0]}, expected {self._n_nodes} nodes")
            if name not in self._fields:
                self._fields.create_dataset(
                    name, shape=(0,) + arr.shape, maxshape=(None,) + arr.shape,
                    dtype="f4", chunks=(1,) + arr.shape,
                    compression=self._compression)
            d = self._fields[name]
            d.resize(i + 1, axis=0)
            d[i] = arr
        d = self._f["iteration"]
        d.resize(i + 1, axis=0)
        d[i] = int(iteration)
        for k, v in (scalars or {}).items():
            if k not in self._scalars:
                # Created on first sight and back-filled with NaN, so a scalar
                # that only appears from some later stage still lines up with
                # the snapshot index.
                ds = self._scalars.create_dataset(
                    k, shape=(i,), maxshape=(None,), dtype="f8", chunks=(64,))
                ds[:] = onp.nan
            ds = self._scalars[k]
            ds.resize(i + 1, axis=0)
            ds[i] = float(v)
        # Any scalar not supplied this round also has to grow, or the columns
        # drift out of step with each other.
        for k in self._scalars:
            ds = self._scalars[k]
            if ds.shape[0] == i:
                ds.resize(i + 1, axis=0)
                ds[i] = onp.nan
        if self._xdmf is not None:
            self._xdmf.write_iteration(
                float(iteration),
                point_infos=[(name, _pad_vector(onp.asarray(arr)))
                             for name, arr in zip(self._names, fields)])
            # XDMFWriter only emits its XML in __exit__, so an interrupted run
            # would leave frames in the HDF5 that ParaView cannot index. Rewrite
            # the index now -- it is small XML, and the alternative is losing the
            # whole history to a Ctrl-C.
            self._xdmf._write_xdmf_xml()
            self._xdmf._h5.flush()
        self._n += 1
        self._f.flush()

    def close(self):
        self._f.attrs["n_snapshots"] = self._n
        self._f.close()
        if self._xdmf is not None:
            self._xdmf.__exit__(None, None, None)
            self._xdmf = None

    @property
    def xdmf_path(self):
        return None if self._xdmf is None else self._xdmf._xdmf_filename

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _as_attr(v):
    """HDF5-storable form of a config value; ``repr`` for anything exotic.

    Sequences of strings are encoded to fixed-width BYTES. h5py cannot store a
    numpy ``<U`` array as an attribute at all ("No conversion path for dtype"),
    and a list of load-case names is exactly the sort of thing a sweep wants to
    read back, so it must not fall through to ``repr``.
    """
    if isinstance(v, (int, float, bool, str, onp.integer, onp.floating)):
        return v
    if isinstance(v, (list, tuple, onp.ndarray)):
        try:
            arr = onp.asarray(v)
        except Exception:
            return repr(v)
        if arr.dtype.kind in "ifb":
            return arr
        if arr.dtype.kind in "US":
            return onp.array([str(x).encode("utf-8") for x in arr.ravel()])
    return repr(v)


def _pad_vector(arr):
    """Per-node field in a layout XDMF accepts.

    A 2-component vector is padded to 3: XDMF takes ``k in {1,2,3,6,9}``, but
    ParaView renders a 2-component attribute as a scalar triple rather than a
    glyph-able vector, so a 2-D director has to be handed over with a zero z.
    """
    arr = onp.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return onp.column_stack([arr, onp.zeros(arr.shape[0], arr.dtype)])
    return arr


_ELE_TYPE = {(2, 3): "TRI3", (2, 4): "QUAD4", (2, 6): "TRI6",
             (3, 4): "TET4", (3, 8): "HEX8", (3, 10): "TET10", (3, 20): "HEX20"}


def snapshots_to_xdmf(h5_path: str, out_path: str = None, fields=None):
    """Convert an existing ``snapshots.h5`` into a ParaView-readable XDMF pair.

    For files written before the writer grew its own XDMF stream, or to
    re-emit one with a different field subset. Returns the ``.xdmf`` path.

    ``fields`` selects which of ``/fields`` to carry over (default: all).
    """
    import h5py
    from feax.mesh import Mesh
    from feax.utils import XDMFWriter

    out_path = out_path or (os.path.splitext(h5_path)[0] + "_evolution.xdmf")
    with h5py.File(h5_path, "r") as h:
        pts = onp.asarray(h["mesh/points"], dtype=onp.float64)
        cells = onp.asarray(h["mesh/cells"], dtype=onp.int32)
        key = (pts.shape[1], cells.shape[1])
        if key not in _ELE_TYPE:
            raise ValueError(f"snapshots_to_xdmf: cannot infer element type "
                             f"from {pts.shape[1]}-D points with "
                             f"{cells.shape[1]} nodes per cell")
        names = list(fields) if fields else list(h["fields"])
        data = {n: onp.asarray(h["fields"][n]) for n in names}
        iters = onp.asarray(h["iteration"])
        mesh = Mesh(pts, cells, ele_type=_ELE_TYPE[key])
        with XDMFWriter(mesh, out_path) as w:
            prev = None
            for i, t in enumerate(iters):
                # Files written before the writer's duplicate guard can repeat
                # the last iteration (the end-of-run snapshot landing on a
                # multiple of the period). XDMF needs strictly increasing times.
                if prev is not None and t <= prev:
                    continue
                prev = t
                w.write_iteration(float(t), point_infos=[
                    (n, _pad_vector(data[n][i])) for n in names])
    return out_path
