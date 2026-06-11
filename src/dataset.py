"""
Data loading layer for supported symbolic regression benchmarks.

Dataset Structure
-----------------
- HuggingFace repository: nnheui/llm-srbench
- Data splits:
    lsr_transform              → TransformedFeynman benchmark
    lsr_synth_bio_pop_growth   → Biological population growth
    lsr_synth_chem_react       → Chemical reaction kinetics
    lsr_synth_matsci           → Materials science
    lsr_synth_phys_osc         → Physical oscillator
- Actual samples stored in HDF5 file lsr_bench_data.hdf5:
    /lsr_synth/{short_name}/{equation_name}/train_data    → X, y
    /lsr_synth/{short_name}/{equation_name}/id_test_data  → X, y
    /lsr_synth/{short_name}/{equation_name}/ood_test_data → X, y (optional)
    /lsr_transform/{equation_name}/train                  → X, y
    /lsr_transform/{equation_name}/test                   → X, y
- AI-Feynman converted local HDF5:
    /feynman/{equation_name}/X, y
    /bonus/{equation_name}/X, y
  A fixed deterministic 10k/10k train/test sample is selected per equation.

Usage Examples
--------------
# Load the first equation from bio_pop_growth
ds = SRDataset.from_srbench("bio_pop_growth")
ds.load()
print(ds.summary())

# Load a specific equation by name
ds = SRDataset.from_srbench("bio_pop_growth", equation_name="lsr_synth_bio_pop_growth_0")
ds.load()

# Build directly from local NumPy arrays (for custom data or unit tests)
ds = SRDataset.from_arrays(X_train, y_train, symbols=["t", "P"], ...)
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np


# Dataset short name → (HuggingFace split suffix, HDF5 path prefix, default variables)
_SPLIT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "bio_pop_growth": {
        "hf_split": "lsr_synth_bio_pop_growth",
        "hdf5_prefix": "lsr_synth/bio_pop_growth",
        "sample_keys": ("train", "test", "ood_test"),
        "default_symbols": ["dP_dt", "t", "P"],
        "default_symbol_descs": [
            "Population growth rate",
            "Time",
            "Population at time t",
        ],
    },
    "chem_react": {
        "hf_split": "lsr_synth_chem_react",
        "hdf5_prefix": "lsr_synth/chem_react",
        "sample_keys": ("train", "test", "ood_test"),
        "default_symbols": ["dA_dt", "t", "A"],
        "default_symbol_descs": [
            "Rate of change of concentration in chemistry reaction kinetics",
            "Time",
            "Concentration at time t",
        ],
    },
    "matsci": {
        "hf_split": "lsr_synth_matsci",
        "hdf5_prefix": "lsr_synth/matsci",
        "sample_keys": ("train", "test", "ood_test"),
        "default_symbols": None,
        "default_symbol_descs": None,
    },
    "phys_osc": {
        "hf_split": "lsr_synth_phys_osc",
        "hdf5_prefix": "lsr_synth/phys_osc",
        "sample_keys": ("train", "test", "ood_test"),
        "default_symbols": ["dv_dt", "x", "t", "v"],
        "default_symbol_descs": [
            "Acceleration in Non-linear Harmonic Oscillator",
            "Position at time t",
            "Time",
            "Velocity at time t",
        ],
    },
    "lsrtransform": {
        "hf_split": "lsr_transform",
        "hdf5_prefix": "lsr_transform",
        "sample_keys": ("train", "test"),
        "default_symbols": None,
        "default_symbol_descs": None,
    },
}

HF_REPO_ID = "nnheui/llm-srbench"
HDF5_FILENAME = "lsr_bench_data.hdf5"
AIFEYNMAN_HDF5_FILENAME = "aifeynman_data.hdf5"
AIFEYNMAN_SAMPLE_SIZE = 10_000

# Local, repo-resident dataset location. Resolving datasets here (instead of the
# HuggingFace ~/.cache download dir) makes runs reproducible and independent of the
# current working directory / per-user HF cache. Override with $SYMREG_DATASETS_DIR.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = Path(os.environ.get("SYMREG_DATASETS_DIR", REPO_ROOT / "datasets"))

SRBENCH_DATASETS = {
    "llm-srbench": DATASETS_ROOT / "llm-srbench",
    "llm-srbench-noise1pct": DATASETS_ROOT / "llm-srbench-noise1pct",
    "llm-srbench-noise5pct": DATASETS_ROOT / "llm-srbench-noise5pct",
}

AIFEYNMAN_SPLITS: Dict[str, Dict[str, str]] = {
    "feynmanequations": {
        "hdf5_group": "feynman",
        "csv": "FeynmanEquations.csv",
    },
    "bonusequations": {
        "hdf5_group": "bonus",
        "csv": "BonusEquations.csv",
    },
}

AIFEYNMAN_SPLIT_ALIASES = {
    "feynman": "feynmanequations",
    "feynmanequations": "feynmanequations",
    "feynmanequations.csv": "feynmanequations",
    "bonusequations": "bonusequations",
    "bonusequations.csv": "bonusequations",
    "bonus": "bonusequations",
}

DATASET_ALIASES = {
    "llm_srbench": "llm-srbench",
    "srbench": "llm-srbench",
    "llmsrbench": "llm-srbench",
    "llm-srbench": "llm-srbench",
    "llm-srbench-noise1": "llm-srbench-noise1pct",
    "llm-srbench-noise-1pct": "llm-srbench-noise1pct",
    "llm-srbench-noise1pct": "llm-srbench-noise1pct",
    "llm-srbench-noise5": "llm-srbench-noise5pct",
    "llm-srbench-noise-5pct": "llm-srbench-noise5pct",
    "llm-srbench-noise5pct": "llm-srbench-noise5pct",
    "aifeynman": "aifeynman",
    "ai-feynman": "aifeynman",
}

SRBENCH_SPLIT_ALIASES = {
    "bio_pop_growth": "bio_pop_growth",
    "biology": "bio_pop_growth",
    "chem_react": "chem_react",
    "chemistry": "chem_react",
    "matsci": "matsci",
    "materials": "matsci",
    "phys_osc": "phys_osc",
    "physics": "phys_osc",
    "lsrtransform": "lsrtransform",
    "lsr_transform": "lsrtransform",
}

# Default llm-srbench snapshot directory (contains lsr_bench_data.hdf5 + data/*.parquet).
DEFAULT_REPO_DIR = SRBENCH_DATASETS["llm-srbench"]

_repo_dir_cache: Dict[str, Path] = {}
_repo_dir_override: Optional[Path] = None


def normalize_dataset_name(dataset_name: Optional[str]) -> str:
    """Normalize public dataset family names accepted by the CLI."""
    key = (dataset_name or "llm-srbench").strip().lower()
    key = DATASET_ALIASES.get(key, key)
    if key not in SRBENCH_DATASETS and key != "aifeynman":
        available = list(SRBENCH_DATASETS.keys()) + ["aifeynman"]
        raise ValueError(f"Unknown dataset='{dataset_name}', available: {available}")
    return key


def normalize_split_name(dataset_name: str, split_name: str) -> str:
    """Normalize a split name within a dataset family."""
    dataset_name = normalize_dataset_name(dataset_name)
    key = split_name.strip()
    if dataset_name == "aifeynman":
        norm = AIFEYNMAN_SPLIT_ALIASES.get(key.lower(), key.lower())
        if norm not in AIFEYNMAN_SPLITS:
            raise ValueError(
                f"Unknown aifeynman split='{split_name}', "
                f"available: {list(AIFEYNMAN_SPLITS.keys())}"
            )
        return norm

    key = SRBENCH_SPLIT_ALIASES.get(key.lower(), key)
    if key not in _SPLIT_REGISTRY:
        raise ValueError(
            f"Unknown llm-srbench split='{split_name}', "
            f"available: {list(_SPLIT_REGISTRY.keys())}"
        )
    return key


def dataset_split_names(dataset_name: str) -> List[str]:
    """Return all split names for a dataset family in stable default order."""
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "aifeynman":
        return list(AIFEYNMAN_SPLITS.keys())
    return list(_SPLIT_REGISTRY.keys())


def _repo_has_expected_files(path: Path) -> bool:
    """Return True when the directory looks like a usable llm-srbench snapshot."""
    data_dir = path / "data"
    if not (path / HDF5_FILENAME).is_file() or not data_dir.is_dir():
        return False
    return any(data_dir.glob("*.parquet"))


def _download_repo_to(path: Optional[Path] = None, quiet: bool = True) -> Path:
    """Download the HF dataset repo, optionally materializing it at a requested path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "Missing dependency: huggingface_hub. Please run: pip install huggingface_hub"
        )

    if path is None:
        if not quiet:
            print(f"[SRDataset] Downloading repository {HF_REPO_ID} ...")
        return Path(snapshot_download(repo_id=HF_REPO_ID, repo_type="dataset"))

    path.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"[SRDataset] Dataset repo not found; downloading {HF_REPO_ID} to {path} ...")
    return Path(
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=str(path),
        )
    )


def get_repo_dir(quiet: bool = True, dataset_name: str = "llm-srbench") -> Path:
    """Return the local path to the HF dataset repo (download once, then reuse cached path).

    Resolution order:
      1. an explicit path previously registered via ``set_repo_dir``;
      2. the repo-local snapshot for the requested llm-srbench family;
      3. otherwise download the clean HF dataset into ``DEFAULT_REPO_DIR``.
    """
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "aifeynman":
        raise ValueError("aifeynman does not use the llm-srbench repo directory")
    if _repo_dir_override is not None:
        return _repo_dir_override
    if dataset_name in _repo_dir_cache:
        return _repo_dir_cache[dataset_name]

    default_dir = SRBENCH_DATASETS[dataset_name]
    if _repo_has_expected_files(default_dir):
        _repo_dir_cache[dataset_name] = default_dir
    elif dataset_name == "llm-srbench":
        _repo_dir_cache[dataset_name] = _download_repo_to(default_dir, quiet=quiet)
    else:
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' is not available at {default_dir}. "
            f"Expected {HDF5_FILENAME} and data/*.parquet."
        )
    return _repo_dir_cache[dataset_name]


def set_repo_dir(path: str, quiet: bool = False) -> None:
    """Set the local dataset repo path, downloading the HF dataset there if missing."""
    global _repo_dir_override
    p = Path(path)
    if not _repo_has_expected_files(p):
        p = _download_repo_to(p, quiet=quiet)
    _repo_dir_override = p


def find_equation_split(equation_name: str, dataset_name: str = "llm-srbench") -> Optional[str]:
    """
    Reverse-lookup the split that contains the given equation name.
    Iterates over all splits' parquet metadata to find the one containing the equation.
    Returns the split short name (e.g. 'matsci'), or None if not found.
    """
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "aifeynman":
        for split_name in dataset_split_names(dataset_name):
            if equation_name in list_equations(dataset_name, split_name):
                return split_name
        return None

    try:
        import pandas as pd
    except ImportError:
        return None

    repo_dir = get_repo_dir(dataset_name=dataset_name)
    for split_name, reg in _SPLIT_REGISTRY.items():
        parquet_files = list((repo_dir / "data").glob(f"{reg['hf_split']}-*.parquet"))
        if not parquet_files:
            continue
        df = pd.read_parquet(parquet_files[0])
        if equation_name in df["name"].values:
            return split_name
    return None


def list_equations(dataset_name: str, split_name: str) -> List[str]:
    """List concrete equation names for one dataset split."""
    dataset_name = normalize_dataset_name(dataset_name)
    split_name = normalize_split_name(dataset_name, split_name)

    if dataset_name == "aifeynman":
        split_info = AIFEYNMAN_SPLITS[split_name]
        csv_path = DATASETS_ROOT / "aifeynman" / split_info["csv"]
        if csv_path.is_file():
            try:
                import pandas as pd

                df = pd.read_csv(csv_path)
                names = df["Filename"].dropna().tolist()
                return [str(x).strip() for x in names if str(x).strip()]
            except Exception:
                pass

        hdf5_path = DATASETS_ROOT / "aifeynman" / AIFEYNMAN_HDF5_FILENAME
        if not hdf5_path.is_file():
            return []
        try:
            import h5py

            with h5py.File(hdf5_path, "r") as f:
                return list(f[split_info["hdf5_group"]].keys())
        except Exception:
            return []

    try:
        import pandas as pd
    except ImportError:
        return []

    reg = _SPLIT_REGISTRY[split_name]
    repo_dir = get_repo_dir(dataset_name=dataset_name)
    parquet_files = list((repo_dir / "data").glob(f"{reg['hf_split']}-*.parquet"))
    if not parquet_files:
        return []
    df = pd.read_parquet(parquet_files[0])
    return df["name"].tolist()


def resolve_requested_cases(
    dataset_name: str,
    split_names: Optional[List[str]] = None,
    equation_names: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Resolve CLI dataset/split/equation selectors to concrete test cases.

    Split-selected cases and explicitly selected equations are unioned in stable
    order: all requested split cases first, then explicit equations not already
    present.  If both split_names and equation_names are empty, all splits in the
    dataset family are selected.
    """
    dataset_name = normalize_dataset_name(dataset_name)
    split_names = [s for s in (split_names or []) if s]
    equation_names = [e for e in (equation_names or []) if e]
    if not split_names and not equation_names:
        split_names = dataset_split_names(dataset_name)

    cases: List[Dict[str, str]] = []
    seen = set()

    for split in split_names:
        split = normalize_split_name(dataset_name, split)
        for eq in list_equations(dataset_name, split):
            key = (dataset_name, split, eq)
            if key not in seen:
                cases.append({"dataset": dataset_name, "split": split, "equation": eq})
                seen.add(key)

    for eq in equation_names:
        split = find_equation_split(eq, dataset_name=dataset_name)
        if split is None:
            raise ValueError(f"Equation '{eq}' not found in dataset '{dataset_name}'")
        key = (dataset_name, split, eq)
        if key not in seen:
            cases.append({"dataset": dataset_name, "split": split, "equation": eq})
            seen.add(key)

    return cases


def _json_attr(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return value


def _stable_aifeynman_indices(
    n_rows: int,
    dataset_key: str,
    sample_size: int = AIFEYNMAN_SAMPLE_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    required = sample_size * 2
    if n_rows < required:
        raise ValueError(
            f"AI-Feynman case has only {n_rows} rows; need at least {required} "
            f"for fixed {sample_size}/{sample_size} train/test samples."
        )
    digest = hashlib.sha256(dataset_key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_rows, size=required, replace=False)
    return np.sort(idx[:sample_size]), np.sort(idx[sample_size:])


def _format_unit_desc(unit_info: Any) -> str:
    if not isinstance(unit_info, dict):
        return ""
    unit_name = str(unit_info.get("Units", "")).strip()
    dims = []
    for key in ("m", "s", "kg", "T", "V"):
        val = unit_info.get(key)
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if abs(fval) > 1e-12:
            dims.append(f"{key}^{fval:g}")
    if dims:
        return f"{unit_name}; dimensions {' '.join(dims)}" if unit_name else " ".join(dims)
    return unit_name


class SRDataset:
    """
    Symbolic regression dataset wrapper. Supports two initialization modes:
    - `SRDataset.from_srbench()`: load from nnheui/llm-srbench HuggingFace dataset
    - `SRDataset.from_arrays()`: build directly from NumPy arrays (local testing / custom tasks)
    """

    def __init__(self):
        self.equation_name: str = ""
        self.dataset_identifier: str = ""
        self.symbols: List[str] = []
        self.symbol_descs: List[str] = []
        self.symbol_properties: List[str] = []
        self.expression: str = ""
        self.units: Dict[str, Any] = {}
        self.ranges: Dict[str, Tuple[float, float]] = {}
        self.full_sample_count: Optional[int] = None

        self.X_train: np.ndarray = np.array([])
        self.y_train: np.ndarray = np.array([])
        self.X_test: np.ndarray = np.array([])
        self.y_test: np.ndarray = np.array([])
        self.X_ood_test: np.ndarray = np.array([])
        self.y_ood_test: np.ndarray = np.array([])

        # Loading parameters for internal load() use
        self._load_mode: str = "array"  # "srbench" | "array"
        self._dataset_name: str = "custom"
        self._split_name: Optional[str] = None
        self._equation_name_filter: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Factory methods
    # ------------------------------------------------------------------ #

    @classmethod
    def from_srbench(
        cls,
        split_name: str,
        equation_name: Optional[str] = None,
        dataset_name: str = "llm-srbench",
    ) -> "SRDataset":
        """
        Create a dataset instance that loads from nnheui/llm-srbench (lazy loading, triggered by .load()).

        Parameters
        ----------
        split_name    : Dataset short name, one of 'bio_pop_growth' | 'chem_react' |
                        'matsci' | 'phys_osc' | 'lsrtransform'
        equation_name : Specify a particular equation name (e.g. 'lsr_synth_bio_pop_growth_0').
                        When None, loads the first equation in the split.
        """
        dataset_name = normalize_dataset_name(dataset_name)
        if dataset_name == "aifeynman":
            return cls.from_aifeynman(split_name, equation_name=equation_name)
        split_name = normalize_split_name(dataset_name, split_name)
        obj = cls()
        obj._load_mode = "srbench"
        obj._dataset_name = dataset_name
        obj._split_name = split_name
        obj._equation_name_filter = equation_name
        obj.dataset_identifier = f"{dataset_name}:{split_name}"
        return obj

    @classmethod
    def from_aifeynman(
        cls,
        split_name: str,
        equation_name: Optional[str] = None,
    ) -> "SRDataset":
        """Create a lazy AI-Feynman dataset view backed by the converted HDF5."""
        split_name = normalize_split_name("aifeynman", split_name)
        obj = cls()
        obj._load_mode = "aifeynman"
        obj._dataset_name = "aifeynman"
        obj._split_name = split_name
        obj._equation_name_filter = equation_name
        obj.dataset_identifier = f"aifeynman:{split_name}"
        return obj

    @classmethod
    def from_benchmark(
        cls,
        dataset_name: str,
        split_name: str,
        equation_name: Optional[str] = None,
    ) -> "SRDataset":
        """Dispatch to the proper backend for a concrete benchmark case."""
        dataset_name = normalize_dataset_name(dataset_name)
        if dataset_name == "aifeynman":
            return cls.from_aifeynman(split_name, equation_name=equation_name)
        return cls.from_srbench(
            split_name,
            equation_name=equation_name,
            dataset_name=dataset_name,
        )

    @classmethod
    def from_arrays(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        symbols: List[str],
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        symbol_descs: Optional[List[str]] = None,
        expression: str = "",
        equation_name: str = "custom",
        dataset_identifier: str = "custom",
    ) -> "SRDataset":
        """Build directly from NumPy arrays, no download needed."""
        obj = cls()
        obj._load_mode = "array"
        obj.equation_name = equation_name
        obj.dataset_identifier = dataset_identifier
        obj.symbols = symbols
        obj.symbol_descs = symbol_descs or [""] * len(symbols)
        obj.expression = expression
        obj.X_train = X_train
        obj.y_train = y_train
        if X_test is not None:
            obj.X_test = X_test
        if y_test is not None:
            obj.y_test = y_test
        return obj

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def load(self, quiet: bool = False) -> None:
        """Trigger data loading. No-op in array mode (data already provided at construction)."""
        if self._load_mode == "srbench":
            self._load_from_srbench(quiet=quiet)
        elif self._load_mode == "aifeynman":
            self._load_from_aifeynman(quiet=quiet)
        # Array mode: no action needed

    def _load_from_srbench(self, quiet: bool = False) -> None:
        """
        Load from local HF dataset repo cache.
        Read metadata from Parquet, read sample data from HDF5.
        """
        try:
            import h5py
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                f"Missing dependency: {e}. Please run: pip install h5py pandas pyarrow"
            )

        split_name = self._split_name
        reg = _SPLIT_REGISTRY[split_name]
        hf_split = reg["hf_split"]
        hdf5_prefix = reg["hdf5_prefix"]
        sample_keys = reg["sample_keys"]

        repo_dir = get_repo_dir(quiet=quiet, dataset_name=self._dataset_name)
        hdf5_path = repo_dir / HDF5_FILENAME

        # Read metadata from parquet files
        parquet_pattern = list((repo_dir / "data").glob(f"{hf_split}-*.parquet"))
        if not parquet_pattern:
            raise FileNotFoundError(
                f"No parquet files found for split '{hf_split}'. "
                f"Directory contents: {list((repo_dir / 'data').iterdir())}"
            )
        meta_df = pd.read_parquet(parquet_pattern[0])
        if not quiet:
            print(
                f"[SRDataset] Loaded metadata: {parquet_pattern[0].name}, "
                f"{len(meta_df)} equations total"
            )

        # Filter target equation
        if self._equation_name_filter:
            matched = meta_df[meta_df["name"] == self._equation_name_filter]
            if matched.empty:
                available = meta_df["name"].tolist()
                raise ValueError(
                    f"Equation '{self._equation_name_filter}' not found.\n"
                    f"Available equations ({len(available)} total): {available[:10]} ..."
                )
            target_row = matched.iloc[0]
        else:
            target_row = meta_df.iloc[0]

        # Parse metadata
        def _get(row, key, fallback=None):
            """Safely extract value from pandas Series, handling None/NaN/empty lists."""
            val = row.get(key)
            if val is None:
                return fallback
            if isinstance(val, float) and np.isnan(val):
                return fallback
            if isinstance(val, (list, np.ndarray)) and len(val) == 0:
                return fallback
            if isinstance(val, np.ndarray):
                return val.tolist()
            return val

        self.equation_name = str(target_row["name"])
        self.symbols = _get(target_row, "symbols", reg.get("default_symbols")) or []
        self.symbol_descs = _get(target_row, "symbol_descs", reg.get("default_symbol_descs")) or []
        self.symbol_properties = _get(target_row, "symbol_properties") or []
        self.expression = _get(target_row, "expression", "") or ""

        # Read samples from HDF5
        eq_path = f"/{hdf5_prefix}/{self.equation_name}"
        if not quiet:
            print(f"[SRDataset] Reading HDF5 path: {eq_path}")

        with h5py.File(hdf5_path, "r") as f:
            eq_grp = f[eq_path]

            train_key = sample_keys[0]
            test_key = sample_keys[1]
            ood_key = sample_keys[2] if len(sample_keys) > 2 else None

            self.X_train, self.y_train = self._read_h5_item(eq_grp[train_key])
            self.X_test, self.y_test = self._read_h5_item(eq_grp[test_key])
            if ood_key and ood_key in eq_grp:
                self.X_ood_test, self.y_ood_test = self._read_h5_item(eq_grp[ood_key])

        self._clip_near_zero_negatives()

        if not quiet:
            print(
                f"[SRDataset] Loading complete: equation={self.equation_name}, "
                f"train={self.X_train.shape}, test={self.X_test.shape}"
            )

    def _load_from_aifeynman(self, quiet: bool = False) -> None:
        """Load a fixed 10k/10k train/test sample from the converted AI-Feynman HDF5."""
        try:
            import h5py
        except ImportError as e:
            raise ImportError(f"Missing dependency: {e}. Please run: pip install h5py")

        split_name = normalize_split_name("aifeynman", self._split_name or "")
        split_info = AIFEYNMAN_SPLITS[split_name]
        hdf5_path = DATASETS_ROOT / "aifeynman" / AIFEYNMAN_HDF5_FILENAME
        if not hdf5_path.is_file():
            raise FileNotFoundError(
                f"AI-Feynman HDF5 not found: {hdf5_path}. "
                "Run datasets/convert_aifeynman_to_hdf5.py first."
            )

        available = list_equations("aifeynman", split_name)
        if not available:
            raise ValueError(f"No equations found for aifeynman split '{split_name}'")
        equation_name = self._equation_name_filter or available[0]
        if equation_name not in available:
            raise ValueError(
                f"Equation '{equation_name}' not found in aifeynman split '{split_name}'. "
                f"Available examples: {available[:10]}"
            )

        group_name = split_info["hdf5_group"]
        if not quiet:
            print(f"[SRDataset] Reading AI-Feynman HDF5: /{group_name}/{equation_name}")

        with h5py.File(hdf5_path, "r") as f:
            eq_grp = f[group_name][equation_name]
            X_ds = eq_grp["X"]
            y_ds = eq_grp["y"]
            n_rows = int(X_ds.shape[0])
            train_idx, test_idx = _stable_aifeynman_indices(
                n_rows,
                dataset_key=f"aifeynman:{split_name}:{equation_name}:v1",
            )

            self.X_train = np.asarray(X_ds[train_idx], dtype=np.float64)
            self.y_train = np.asarray(y_ds[train_idx], dtype=np.float64).ravel()
            self.X_test = np.asarray(X_ds[test_idx], dtype=np.float64)
            self.y_test = np.asarray(y_ds[test_idx], dtype=np.float64).ravel()

            attrs = dict(eq_grp.attrs)

        feature_names = _json_attr(attrs.get("feature_names"), [])
        output_name = str(attrs.get("output", "y"))
        self.equation_name = equation_name
        self.dataset_identifier = f"aifeynman:{split_name}"
        self.symbols = [output_name] + list(feature_names)
        self.units = _json_attr(attrs.get("variable_units"), {})
        self.ranges = _json_attr(attrs.get("ranges"), {})
        self.expression = str(attrs.get("formula", ""))
        self.full_sample_count = n_rows
        self.symbol_descs = [
            _format_unit_desc(self.units.get(symbol, {}))
            for symbol in self.symbols
        ]
        self.symbol_properties = []
        self.X_ood_test = np.array([])
        self.y_ood_test = np.array([])

        if not quiet:
            print(
                f"[SRDataset] Loading complete: equation={self.equation_name}, "
                f"full={self.full_sample_count}, train={self.X_train.shape}, "
                f"test={self.X_test.shape}"
            )

    def _clip_near_zero_negatives(self, eps: float = 1e-6) -> None:
        """Clip very small negative values (> -eps) in feature matrices to 0, eliminating numerical noise from ODE solvers."""
        for arr in (self.X_train, self.X_test, self.X_ood_test):
            if arr is not None and arr.size > 0:
                mask = (arr < 0) & (arr > -eps)
                if mask.any():
                    arr[mask] = 0.0

    @staticmethod
    def _read_h5_item(item) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read X and y from an HDF5 Dataset or Group.

        srbench actual storage format is a 2D array (n_samples, n_cols):
          - First column is the target variable y (e.g. dP_dt)
          - Remaining columns are feature variables X (e.g. t, P)
        """
        import h5py

        if isinstance(item, h5py.Dataset):
            data = np.array(item, dtype=np.float64)
            # First column = y, remaining columns = X
            y = data[:, 0].ravel()
            X = data[:, 1:]
            return X, y

        # Compatible with Group format (containing 'X'/'y' sub-fields)
        keys = list(item.keys())
        if "X" in keys and "y" in keys:
            X = np.array(item["X"], dtype=np.float64)
            y = np.array(item["y"], dtype=np.float64).ravel()
            return X, y

        # Fallback: concatenate all fields, treat first column as y
        arrays = [np.array(item[k], dtype=np.float64) for k in sorted(keys)]
        stacked = np.column_stack([a.ravel() if a.ndim == 1 else a for a in arrays])
        y = stacked[:, 0].ravel()
        X = stacked[:, 1:]
        return X, y

    # ------------------------------------------------------------------ #
    # Utility methods
    # ------------------------------------------------------------------ #

    @property
    def feature_names(self) -> List[str]:
        """
        List of feature variable names for SymPy lambdify (excluding the target variable, i.e. the first symbol).
        For lsr_synth datasets, symbols[0] is the target variable (e.g. dP_dt),
        symbols[1:] are feature variables (e.g. ['t', 'P']).
        The number of columns is determined by the actual X_train column count.
        """
        if len(self.symbols) > 1 and self.X_train.ndim == 2:
            n_features = self.X_train.shape[1]
            # symbols[0] is y, symbols[1:] are X columns
            feature_syms = self.symbols[1:]
            if len(feature_syms) == n_features:
                return feature_syms
        # Fallback: use symbols directly
        if self.X_train.ndim == 2:
            return self.symbols[: self.X_train.shape[1]]
        return self.symbols

    def get_context_prompt(self) -> str:
        """Return a formatted physical context description for LLM to understand the current regression task.
        Note: does not include self.expression (ground truth) to avoid leaking the answer."""
        feat_names = self.feature_names
        lines = [
            f"Task: {self.equation_name} ({self.dataset_identifier})",
            f"Target variable: {self.symbols[0] if self.symbols else '?'}",
            f"Feature variables: {feat_names}",
        ]
        if self.symbol_descs:
            desc_pairs = list(zip(self.symbols, self.symbol_descs))
            desc_str = ", ".join(f"{s}={d}" for s, d in desc_pairs if d)
            if desc_str:
                label = "Dimensional information" if self._dataset_name == "aifeynman" else "Physical meaning"
                lines.append(f"{label}: {desc_str}")
        if self._dataset_name == "aifeynman":
            lines.append(
                "Task metadata: use the dimensional information, variable names, "
                "feature ranges, and sampled numerical behavior."
            )
            if self.ranges:
                range_items = []
                for name in feat_names:
                    bounds = self.ranges.get(name)
                    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                        range_items.append(f"{name} in [{bounds[0]}, {bounds[1]}]")
                if range_items:
                    lines.append(f"Feature ranges: {', '.join(range_items)}")
            if self.full_sample_count is not None:
                lines.append(
                    f"Sampling: fixed deterministic uniform sample of "
                    f"{len(self.y_train)} train and {len(self.y_test)} test rows "
                    f"from {self.full_sample_count} generated rows."
                )
        lines.append(f"Training samples: {len(self.y_train)}")
        return "\n".join(lines)

    def summary(self) -> str:
        """Return a dataset summary string."""
        lines = [
            f"=== Dataset: {self.equation_name} ({self.dataset_identifier}) ===",
            f"Symbols     : {self.symbols}",
            f"Features    : {self.feature_names}",
            f"Reference   : {self.expression or 'unknown'}",
            f"Train set   : X={self.X_train.shape}, y={self.y_train.shape}",
        ]
        if self.X_test.size > 0:
            lines.append(f"Test set    : X={self.X_test.shape}, y={self.y_test.shape}")
        if self.X_ood_test.size > 0:
            lines.append(f"OOD test set: X={self.X_ood_test.shape}, y={self.y_ood_test.shape}")
        if self.full_sample_count is not None:
            lines.append(f"Full source : {self.full_sample_count} rows")
        return "\n".join(lines)

    def list_equations(self) -> List[str]:
        """
        List all available equation names in the current split.
        """
        if self._split_name is None:
            return []
        return list_equations(self._dataset_name, self._split_name)
