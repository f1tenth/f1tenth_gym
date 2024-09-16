from typing import Any, Dict, TypeVar

KeyType = TypeVar("KeyType")


def deep_update(
    mapping: Dict[KeyType, Any], *updating_mappings: Dict[KeyType, Any]
) -> Dict[KeyType, Any]:
    """Recursive update of dict from pydantic: https://github.com/pydantic/pydantic/blob/fd2991fe6a73819b48c906e3c3274e8e47d0f761/pydantic/utils.py#L200

    Parameters
    ----------
    mapping : Dict[KeyType, Any]
        mapping of original dict

    Returns
    -------
    Dict[KeyType, Any]
        updated dict
    """
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping
