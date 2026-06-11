import numpy as np
from src.engine.benchmark import ModelBenchmark, BenchmarkCandidate


class ConstModel(BenchmarkCandidate):
    def __init__(self, name, cls):
        self.name = name
        self.num_params = 1000
        self.size_mb = 0.1
        self._cls = cls

    def predict(self, X):
        return np.full(len(X), self._cls, dtype=int)


def test_run_builds_dataframe_with_expected_columns():
    X = np.zeros((4, 3), dtype=np.float32)
    y = np.array([0, 0, 1, 1])
    bench = ModelBenchmark()
    df = bench.run([ConstModel("always0", 0), ConstModel("always1", 1)], X, y)
    assert list(df["name"]) == ["always0", "always1"]
    for col in ("accuracy", "latency_ms", "num_params", "size_mb"):
        assert col in df.columns
    assert abs(df.loc[df["name"] == "always0", "accuracy"].iloc[0] - 0.5) < 1e-9


def test_to_report_is_markdown_table():
    X = np.zeros((2, 3), dtype=np.float32)
    y = np.array([0, 1])
    bench = ModelBenchmark()
    df = bench.run([ConstModel("m", 0)], X, y)
    md, _plots = bench.to_report(df)
    assert "| name" in md and "accuracy" in md
