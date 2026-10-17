# CPU bundle tools

`contract.py` defines the independent semantic operators and W4A16 codec.
`bundle.py` creates or verifies represented input/reference files and their
hash manifest. `suite_records.py` creates the CPU-only family identity and
declared FP32-policy precheck records. Tests cover both contracts and reader
integrity. For native execution, see `../README.md` and the CUDA runner source
under `artifacts/kernels/volta-grouped-lowbit/variants/`.

Run the checks without regenerating the large FP64 references:

```sh
python3 -m unittest discover -s benchmark_runs/volta-grouped-lowbit/source/tests -v
python3 benchmark_runs/volta-grouped-lowbit/source/bundle.py verify benchmark_runs/volta-grouped-lowbit/cases/fixture_bundle
```

`bundle.py generate <directory>` is an explicit deterministic regeneration
operation. It writes represented inputs and references as well as their
manifests; native result fields must never be written into the CPU oracle
wrapper.
