"""Every top-level package must be importable."""


def test_packages_importable():
    import data  # noqa: F401
    import models  # noqa: F401
    import symmetry  # noqa: F401
    import metrics  # noqa: F401
    import training  # noqa: F401
