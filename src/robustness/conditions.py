from dataclasses import dataclass, field
from typing import Any, Dict, Optional

PIPELINE_VERSION = "robustness_v0.1"


@dataclass(frozen=True)
class ConditionSpec:
    family: str
    name: str
    severity: int
    params: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None

    @property
    def condition_id(self):
        return f"{self.family}__{self.name}"


def severity_rank(condition: ConditionSpec) -> int:
    return int(condition.severity)


def identity_condition():
    return ConditionSpec(family="identity", name="clean", severity=0, params={})


def all_training_conditions():
    conditions = [identity_condition()]

    conditions.append(
        ConditionSpec(
            family="resample",
            name="down_up_16k",
            severity=1,
            params={"intermediate_sr": 16000},
        )
    )

    for bitrate in (128, 64, 32):
        severity = {128: 2, 64: 3, 32: 4}[bitrate]
        conditions.append(
            ConditionSpec(
                family="mp3_codec",
                name=f"mp3_{bitrate}k",
                severity=severity,
                params={"bitrate_kbps": bitrate},
            )
        )

    for snr_db in (20, 10, 5):
        severity = {20: 2, 10: 3, 5: 4}[snr_db]
        conditions.append(
            ConditionSpec(
                family="additive_noise",
                name=f"snr_{snr_db}db",
                severity=severity,
                params={"snr_db": snr_db},
            )
        )

    conditions.append(
        ConditionSpec(
            family="telephony",
            name="bandlimit_300_3400",
            severity=3,
            params={"low_hz": 300, "high_hz": 3400},
        )
    )

    return conditions


def conditions_for_max_severity(max_severity: int):
    return [c for c in all_training_conditions() if c.severity <= max_severity]
