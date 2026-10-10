"""Shared HFSS treatment records; canonical definitions retain their ownership."""

from .common import LumpedRlc, LumpedTerminalPort, TerminalPort
from .hfss import HfssDrivenGeometrySpec

__all__ = ["LumpedRlc", "LumpedTerminalPort", "TerminalPort", "HfssDrivenGeometrySpec"]
