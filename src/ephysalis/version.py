import importlib.metadata


def get_version_from_pyproject():
    return importlib.metadata.version("ephysalis")


# Example usage:
# version = get_version_from_pyproject()
