# Data Format

We standardize the data format to be a CSV file, each line containing
the following:

```
NODE A, NODE B, (NODE A < NODE B?)
```

If `NODE A` is judged to be worse than `NODE B`, then the field is
populated with `1`. Else `0`.
