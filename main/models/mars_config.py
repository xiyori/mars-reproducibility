from dataclasses import dataclass

@dataclass
class MarsConfig:
    enabled: bool = True
    d: int = 3
    tt_rank: int = 8
    auto_shapes: bool = True
    shape: tuple = None
