from packaging import version


def version_lessthan(module, number):
    return version.parse(module.__version__) < version.parse(number)


def version_greaterthan(module, number):
    return version.parse(module.__version__) > version.parse(number)


def version_lessthanorequal(module, number):
    return version.parse(module.__version__) <= version.parse(number)


def version_greaterthanorequal(module, number):
    return version.parse(module.__version__) >= version.parse(number)
