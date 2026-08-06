# minimal stub: only Batch is imported by denoiser (type hint / training use only)
from dataclasses import dataclass
from typing import Any
@dataclass
class Batch:
    obs: Any = None
    act: Any = None
    mask_padding: Any = None
    rew: Any = None
    end: Any = None
    trunc: Any = None
    info: Any = None
