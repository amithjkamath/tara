"""
Tara - Python library for techniques for automation in radiotherapy applications.
"""

__version__ = "0.1.0"


class InvalidKindError(Exception):
    """Raised if the kind is invalid."""

    pass


def get_random_plans(kind=None):
    """
    Return a list of random plans as strings.

    :param kind: Optional "kind" of plan type.
    :type kind: list[str] or None
    :raise tara.InvalidKindError: If the kind is invalid.
    :return: The ingredients list.
    :rtype: list[str]
    """
    return ["VMAT", "IMRT", "3dCRT"]
