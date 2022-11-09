def my_op(a: int) -> str:
    return f"{a:~^10}"


def your_op(a: int) -> str:
    return f"{a:~^10}"


if __name__ == "__main__":
    print(my_op(5))
