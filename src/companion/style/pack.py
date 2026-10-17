import hashlib
from collections import defaultdict

from companion.contracts import StylePack
from companion.privacy import SafeCandidatePackage
from companion.style.models import StyleFeatures


def build_style_pack(
    package: SafeCandidatePackage,
    features: StyleFeatures,
) -> StylePack:
    digest = hashlib.sha256(package.model_dump_json().encode("utf-8")).hexdigest()[:12]
    grouped: dict[str, list[str]] = defaultdict(list)
    for candidate in package.candidates:
        values = grouped[candidate.category.value]
        if len(values) < 2:
            values.append(candidate.text)
    examples = [text for category in sorted(grouped) for text in grouped[category]][:12]
    return StylePack(
        version=f"dev-style-{digest}",
        statistics=features.statistics,
        scene_tags=features.scene_tags,
        forbidden_traits=[
            "do_not_impersonate_owner",
            "do_not_invent_private_experiences",
            "do_not_mechanically_repeat",
            "do_not_guess_missing_category_style",
        ],
        examples=examples,
    )
