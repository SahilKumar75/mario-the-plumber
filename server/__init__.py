# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline Doctor environment server components."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_COMPAT_MODULE = None


def _load_compat_module():
    global _COMPAT_MODULE
    if _COMPAT_MODULE is not None:
        return _COMPAT_MODULE

    root_server_path = Path(__file__).resolve().parents[1] / "server.py"
    spec = spec_from_file_location("mario_root_server", root_server_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load compatibility server module from {root_server_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    _COMPAT_MODULE = module
    return _COMPAT_MODULE

from .pipeline_doctor_environment import PipelineDoctorEnvironment


def __getattr__(name: str):
    if name in {"app", "main"}:
        compat_module = _load_compat_module()
        return getattr(compat_module, name)
    raise AttributeError(f"module 'server' has no attribute '{name}'")

__all__ = ["PipelineDoctorEnvironment", "app", "main"]
