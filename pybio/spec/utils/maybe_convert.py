from pybio.spec.utils.maybe_convert_to_v0_3 import maybe_convert_to_v0_3


def maybe_convert(data):
    data = maybe_convert_to_v0_3(data)

    return data
