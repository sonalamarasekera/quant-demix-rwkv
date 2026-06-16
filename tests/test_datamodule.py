import torch

from rwkv_ss.data.datasets.libri2mix import (
    Libri2MixDataset,
    Libri2MixEvalDataset,
    collate_eval_fn,
    collate_train_fn,
)
from rwkv_ss.data.datamodule import Libri2MixDataModule


def test_libri2mix_dataset_segment_shape(synthetic_csv):
    ds = Libri2MixDataset(
        str(synthetic_csv),
        sample_rate=16000,
        segment_seconds=0.25,
    )
    sample = ds[0]

    assert sample["mix"].shape == (1, 4000)
    assert sample["sources"].shape == (2, 1, 4000)


def test_collate_train_fn_pads_to_max_length():
    batch = [
        {"mix": torch.zeros(1, 100), "sources": torch.zeros(2, 1, 100)},
        {"mix": torch.zeros(1, 200), "sources": torch.zeros(2, 1, 200)},
    ]
    mix, sources = collate_train_fn(batch)

    assert mix.shape == (2, 1, 200)
    assert sources.shape == (2, 2, 1, 200)


def test_eval_dataset_returns_row_metadata(synthetic_csv):
    ds = Libri2MixEvalDataset(str(synthetic_csv), sample_rate=16000)
    sample = ds[0]

    assert "row" in sample
    assert "mix_path" in sample["row"]


def test_collate_eval_fn_returns_rows():
    batch = [
        {
            "mix": torch.zeros(1, 100),
            "sources": torch.zeros(2, 1, 100),
            "row": {"mix_path": "a", "s1_path": "b", "s2_path": "c"},
        }
    ]
    mix, sources, rows = collate_eval_fn(batch)

    assert mix.shape[0] == 1
    assert sources.shape[0] == 1
    assert rows[0]["mix_path"] == "a"


def test_datamodule_setup_and_loaders(datamodule_config):
    dm = Libri2MixDataModule(datamodule_config)
    dm.setup(test_csv=datamodule_config.train_csv)

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    mix, sources = next(iter(train_loader))
    assert mix.ndim == 3
    assert sources.ndim == 4

    mix_val, sources_val = next(iter(val_loader))
    assert mix_val.shape[0] <= datamodule_config.batch_size

    mix_test, sources_test, rows = next(iter(test_loader))
    assert len(rows) == mix_test.shape[0]


def test_datamodule_setup_raises_for_missing_csv(datamodule_config, tmp_path):
    dm = Libri2MixDataModule(datamodule_config)
    datamodule_config.train_csv = str(tmp_path / "missing.csv")

    try:
        dm.setup()
        raised = False
    except FileNotFoundError:
        raised = True

    assert raised
