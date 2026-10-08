from dataclasses import dataclass
import math

@dataclass(frozen=True)
class DoseEvent:
    time: float
    amount: float
    route: str
    duration: float = 0.

    def __post_init__(self):
        if not all(math.isfinite(x) for x in [self.time,self.amount,self.duration]):
            raise ValueError('Nonfinite event')
        if self.amount <= 0 or self.duration < 0:
            raise ValueError('Invalid amount/duration')
        if self.route not in {'oral','iv_bolus','iv_infusion'}:
            raise ValueError('Unsupported route')
        if (self.route == 'iv_infusion') != (self.duration > 0):
            raise ValueError('Infusion requires duration; other routes require zero duration')
