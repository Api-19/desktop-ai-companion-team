import re

from companion.privacy.style_candidate_approval import ApprovalWitness


def _validated_witness(value: object, candidate_count: int) -> ApprovalWitness:
    if (
        type(value) is not ApprovalWitness
        or type(value.package_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", value.package_digest) is None
        or type(value.revision) is not str
        or re.fullmatch(r"v1-[0-9a-f]{64}", value.revision) is None
        or type(value.approved_indices) is not tuple
        or not value.approved_indices
        or any(type(index) is not int for index in value.approved_indices)
        or value.approved_indices != tuple(sorted(set(value.approved_indices)))
        or any(index < 0 or index >= candidate_count for index in value.approved_indices)
    ):
        raise ValueError("approval witness unavailable")
    return value
