import matplotlib.pyplot as plt
import numpy as np


def summarize(a: np.ndarray, name: str):
    print(f"{name} shape:", a.shape)
    unique_values, counts = np.unique(a, return_counts=True)
    print(f"{name} unique values:", unique_values)
    print(f"{name} unique counts:", counts)
    print()


if __name__ == "__main__":
    original_output = np.load("output/test_output.npy")
    bioimageio_output = np.load(
        "output/export_test/actual_output_labels_pytorch_state_dict.npy"
    )

    summarize(original_output, "original")
    summarize(bioimageio_output, "bioimageio")
    diff = original_output != bioimageio_output
    summarize(diff, "diff")
    f = plt.figure()
    s = f.add_subplot(111)
    s.imshow(original_output.squeeze())
    s.set_title("original output")

    f = plt.figure()
    s = f.add_subplot(111)
    s.imshow(bioimageio_output.squeeze())
    s.set_title("bioimageio output")

    f = plt.figure()
    s = f.add_subplot(111)
    s.imshow(diff.squeeze())
    s.set_title("diff")
    plt.show()
