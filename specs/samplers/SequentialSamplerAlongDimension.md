example:
```python
from pybio.readers.dummy import DummyReader
from pybio.samplers.sequential import SequentialSamplerAlongDimension

reader = DummyReader()
sampler = SequentialSamplerAlongDimension(sample_dimensions=[0, 0], reader=reader)
print("reader shape:", reader.shape)
for i, s in enumerate(sampler):
    print(f"sample {i:-2}:", [ss.shape for ss in s])
```
Output:
```
reader shape: ((15, 4), (15,))
sample  0 [(1, 4), (1,)]
sample  1 [(1, 4), (1,)]
sample  2 [(1, 4), (1,)]
sample  3 [(1, 4), (1,)]
sample  4 [(1, 4), (1,)]
sample  5 [(1, 4), (1,)]
sample  6 [(1, 4), (1,)]
sample  7 [(1, 4), (1,)]
sample  8 [(1, 4), (1,)]
sample  9 [(1, 4), (1,)]
sample 10 [(1, 4), (1,)]
sample 11 [(1, 4), (1,)]
sample 12 [(1, 4), (1,)]
sample 13 [(1, 4), (1,)]
sample 14 [(1, 4), (1,)]
```