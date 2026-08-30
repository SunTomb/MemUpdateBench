from __future__ import annotations

from types import MappingProxyType

from mub.vnext.contracts.post_core_data import (
    POST_CORE_DATA_SURFACE_CATALOG_VERSION,
    PostCoreAttribute,
    PostCoreDomain,
    PostCoreFamilyId,
    PostCoreSurfaceDeclaration,
)


POST_CORE_SURFACE_CATALOG_VERSION = POST_CORE_DATA_SURFACE_CATALOG_VERSION
POST_CORE_SURFACES = (
    PostCoreSurfaceDeclaration(locale="en-US", surface_id="explicit_canonical"),
    PostCoreSurfaceDeclaration(locale="en-US", surface_id="concise_natural"),
    PostCoreSurfaceDeclaration(locale="es-ES", surface_id="concise_natural"),
    PostCoreSurfaceDeclaration(locale="ja-JP", surface_id="concise_natural"),
)
POST_CORE_SURFACE_KEYS = tuple(surface.surface_key for surface in POST_CORE_SURFACES)

POST_CORE_DOMAIN_IDS: tuple[PostCoreDomain, ...] = (
    "personal",
    "work",
    "community",
    "services",
    "education",
    "travel",
    "household",
    "software",
    "finance",
    "health",
    "media",
    "civic",
)
POST_CORE_ATTRIBUTE_IDS: tuple[PostCoreAttribute, ...] = (
    "location",
    "company",
    "preference",
    "language",
    "timezone",
    "hobby",
    "instrument",
    "project",
    "role",
    "status",
    "priority",
    "contact_method",
)
POST_CORE_FAMILY_IDS: tuple[PostCoreFamilyId, ...] = (
    "interleaved_multi_slot_update",
    "entity_attribute_grounding",
    "noop_write_discipline",
)

# Each family owns four domains. The partition deliberately covers all twelve
# domains once, keeping domain and family effects separable in later releases.
POST_CORE_FAMILY_DOMAIN_MATRIX = MappingProxyType(
    {
        "interleaved_multi_slot_update": (
            "work",
            "education",
            "software",
            "services",
        ),
        "entity_attribute_grounding": (
            "personal",
            "community",
            "media",
            "civic",
        ),
        "noop_write_discipline": (
            "finance",
            "health",
            "travel",
            "household",
        ),
    }
)


__all__ = [
    "POST_CORE_ATTRIBUTE_IDS",
    "POST_CORE_DATA_SURFACE_CATALOG_VERSION",
    "POST_CORE_DOMAIN_IDS",
    "POST_CORE_FAMILY_DOMAIN_MATRIX",
    "POST_CORE_FAMILY_IDS",
    "POST_CORE_SURFACE_CATALOG_VERSION",
    "POST_CORE_SURFACE_KEYS",
    "POST_CORE_SURFACES",
]
