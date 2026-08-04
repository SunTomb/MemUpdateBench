from __future__ import annotations

from dataclasses import dataclass


SurfaceTemplateSet = tuple[str, str, str, str, str, str, str, str, str, str]
ReferenceQueryTemplateSet = tuple[str, str, str, str]


@dataclass(frozen=True, slots=True)
class SurfaceCatalog:
    catalog_version: str
    template_sets: tuple[SurfaceTemplateSet, ...]
    reference_query_template_sets: tuple[ReferenceQueryTemplateSet, ...]
    speakers: tuple[str, ...]
    source_namespace: str
    task_tag: str
    normalization_version: str
    split_policy_version: str

    @property
    def surface_ids(self) -> tuple[str, ...]:
        return tuple(template_set[0] for template_set in self.template_sets)

    @property
    def surface_count(self) -> int:
        return len(self.template_sets)


__all__ = [
    "ReferenceQueryTemplateSet",
    "SurfaceCatalog",
    "SurfaceTemplateSet",
]
