"""Blosc container codec -- MATLAB-side surface only.

Mirrors ``+ndr/+format/+blosc/`` in NDR-matlab. The MATLAB package
exists so callers there can encode/decode Blosc v1 containers without
a system ``zstd`` binary; under the hood it calls ``numcodecs.Blosc``
in a private subprocess venv.

There is no Python mirror. Python callers should use ``numcodecs``
directly:

    from numcodecs import Blosc
    codec = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)
    container = codec.encode(numpy_buffer)
    raw = codec.decode(container)

See ``ndr_matlab_python_bridge.yaml`` in this directory for the
per-function contract.
"""
