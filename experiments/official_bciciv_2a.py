"""Explicit local loader for the official BCI Competition IV 2a GDF release.

MOABB's BNCI mirror is convenient, but an unavailable mirror must not tempt an
experiment to silently change datasets.  This module reads only the official
BCI Competition IV 2a archives and exposes the same trial array contract used
by :mod:`experiments.paper_runner` for ``BNCI2014_001``.

The preprocessing intentionally mirrors ``moabb.paradigms.MotorImagery``:
22 EEG channels, 8--32 Hz band-pass filtering, and the four seconds following
the class cue.  The training GDF carries class events; evaluation labels are
read from the official ``true_labels.zip`` archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
import zlib
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


OFFICIAL_GDF_URL = (
    "https://www.bbci.de/competition/download/competition_iv/"
    "BCICIV_2a_gdf.zip"
)
OFFICIAL_LABELS_URL = "https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip"

GDF_ARCHIVE_NAME = "BCICIV_2a_gdf.zip"
LABEL_ARCHIVE_NAME = "true_labels.zip"
EXPECTED_SUBJECTS = tuple(range(1, 10))
EXPECTED_TRIALS_PER_SESSION = 288
EXPECTED_CHANNELS = 22
EXPECTED_CLASSES = 4
EXPECTED_CLASS_TRIALS_PER_SESSION = 72
SAMPLE_RATE_HZ = 250
EPOCH_SECONDS = 4
EXPECTED_SAMPLES = SAMPLE_RATE_HZ * EPOCH_SECONDS + 1

# MNE's GDF reader preserves the official labels below, but currently reports
# all 25 signal channels as ``eeg``.  The last three signals are EOG and must
# never enter the motor-imagery model.  Keep the full contract explicit rather
# than relying on inferred channel types.
OFFICIAL_EEG_CHANNEL_NAMES = (
    "EEG-Fz",
    "EEG-0",
    "EEG-1",
    "EEG-2",
    "EEG-3",
    "EEG-4",
    "EEG-5",
    "EEG-C3",
    "EEG-6",
    "EEG-Cz",
    "EEG-7",
    "EEG-C4",
    "EEG-8",
    "EEG-9",
    "EEG-10",
    "EEG-11",
    "EEG-12",
    "EEG-13",
    "EEG-14",
    "EEG-Pz",
    "EEG-15",
    "EEG-16",
)
OFFICIAL_EOG_CHANNEL_NAMES = ("EOG-left", "EOG-central", "EOG-right")
OFFICIAL_GDF_CHANNEL_LAYOUT = OFFICIAL_EEG_CHANNEL_NAMES + OFFICIAL_EOG_CHANNEL_NAMES

# BCI Competition IV 2a event codes.  The four class events occur at the cue.
CLASS_EVENT_CODES = {"769": 0, "770": 1, "771": 2, "772": 3}


class OfficialDatasetError(RuntimeError):
    """Raised when the official archive is absent or internally inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _event_code(description: object) -> str:
    """Normalize MNE's numeric GDF annotation descriptions to decimal text."""

    text = str(description).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def official_eeg_picks(channel_names: Iterable[str]) -> list[int]:
    """Return the ordered 22-EEG selection for an official 2a GDF header.

    This deliberately does not use :func:`mne.pick_types`: MNE 1.6.1 labels
    the three known EOG signals as EEG in these files.  An exact layout check
    also prevents a malformed or different GDF release from silently becoming
    paper evidence under the same dataset identifier.
    """

    names = tuple(str(name) for name in channel_names)
    if names != OFFICIAL_GDF_CHANNEL_LAYOUT:
        expected = ", ".join(OFFICIAL_GDF_CHANNEL_LAYOUT)
        observed = ", ".join(names)
        raise OfficialDatasetError(
            "official BCI Competition IV 2a channel layout is invalid; "
            f"expected [{expected}], observed [{observed}]"
        )
    return list(range(EXPECTED_CHANNELS))


class OfficialBCICIV2aSource:
    """In-memory trial source backed by the official 2a GDF and label archives."""

    dataset_id = "BNCI2014_001"
    subject_list = list(EXPECTED_SUBJECTS)

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.extract_root = self.root / "extracted"
        self._subject_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._archive_digest_cache: dict[Path, tuple[int, int, str]] = {}
        self._trial_cache_key_cache: str | None = None

    @property
    def gdf_archive(self) -> Path:
        return self.root / GDF_ARCHIVE_NAME

    @property
    def labels_archive(self) -> Path:
        return self.root / LABEL_ARCHIVE_NAME

    def _require_archive(self, path: Path, url: str) -> None:
        if not path.is_file() or path.stat().st_size == 0:
            raise OfficialDatasetError(
                f"official archive is missing: {path}. Download it from {url} "
                "and pass --data-source official-gdf explicitly."
            )

    @staticmethod
    def _member_for_basename(archive: zipfile.ZipFile, basename: str) -> zipfile.ZipInfo:
        matches = [
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and Path(info.filename).name == basename
        ]
        if len(matches) != 1:
            raise OfficialDatasetError(
                f"archive {archive.filename} must contain exactly one {basename}; "
                f"found {matches!r}"
            )
        return archive.getinfo(matches[0])

    def _archive_sha256(self, path: Path) -> str:
        """Hash an archive once per observed file version within this process."""

        self._require_archive(
            path,
            OFFICIAL_GDF_URL if path == self.gdf_archive else OFFICIAL_LABELS_URL,
        )
        stat = path.stat()
        cached = self._archive_digest_cache.get(path)
        version = (stat.st_size, stat.st_mtime_ns)
        if cached is not None and cached[:2] == version:
            return cached[2]
        digest = _sha256_file(path)
        self._archive_digest_cache[path] = (*version, digest)
        return digest

    @staticmethod
    def _matches_cached_member(path: Path, member: zipfile.ZipInfo) -> bool:
        """Validate an extracted member against its archive metadata."""

        try:
            if not path.is_file() or path.stat().st_size != member.file_size:
                return False
            crc = 0
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    crc = zlib.crc32(chunk, crc)
            return (crc & 0xFFFFFFFF) == member.CRC
        except OSError:
            return False

    def _extract_member(self, archive_path: Path, basename: str) -> Path:
        archive_digest = self._archive_sha256(archive_path)
        destination = self.extract_root / archive_digest / basename
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            member = self._member_for_basename(archive, basename)
            if self._matches_cached_member(destination, member):
                return destination

            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{basename}.",
                    suffix=".part",
                    dir=destination.parent,
                    delete=False,
                ) as target:
                    temporary = Path(target.name)
                    with archive.open(member) as source:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                if not self._matches_cached_member(temporary, member):
                    raise OfficialDatasetError(
                        f"extracted {basename} does not match archive member metadata"
                    )
                temporary.replace(destination)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
        return destination

    def _gdf_path(self, subject: int, session: str) -> Path:
        return self._extract_member(self.gdf_archive, f"A{subject:02d}{session}.gdf")

    def _evaluation_labels(self, subject: int) -> np.ndarray:
        from scipy.io import loadmat

        path = self._extract_member(self.labels_archive, f"A{subject:02d}E.mat")
        mat = loadmat(path)
        if "classlabel" not in mat:
            raise OfficialDatasetError(f"official label file has no classlabel: {path}")
        labels = np.asarray(mat["classlabel"]).reshape(-1).astype(np.int64)
        if labels.size != EXPECTED_TRIALS_PER_SESSION:
            raise OfficialDatasetError(
                f"evaluation labels for subject {subject} have {labels.size} trials; "
                f"expected {EXPECTED_TRIALS_PER_SESSION}"
            )
        if set(labels.tolist()) != set(range(1, EXPECTED_CLASSES + 1)):
            raise OfficialDatasetError(
                f"evaluation labels for subject {subject} are not the expected 1..{EXPECTED_CLASSES}"
            )
        labels -= 1
        self._validate_label_balance(labels, subject, "evaluation")
        return labels

    @staticmethod
    def _validate_label_balance(labels: np.ndarray, subject: int, session: str) -> None:
        counts = Counter(np.asarray(labels, dtype=int).tolist())
        expected = {label: EXPECTED_CLASS_TRIALS_PER_SESSION for label in range(EXPECTED_CLASSES)}
        if counts != expected:
            raise OfficialDatasetError(
                f"{session} label balance for subject {subject} is {dict(counts)}, "
                f"expected {expected}"
            )

    def _read_gdf(self, path: Path, subject: int, session: str) -> tuple[np.ndarray, np.ndarray]:
        import mne

        raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
        picks = official_eeg_picks(raw.ch_names)
        if not np.isclose(raw.info["sfreq"], SAMPLE_RATE_HZ):
            raise OfficialDatasetError(
                f"{path.name} has sample rate {raw.info['sfreq']}; expected {SAMPLE_RATE_HZ}"
            )

        raw.filter(8.0, 32.0, picks=picks, verbose="ERROR")
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        description_by_event = {event: _event_code(description) for description, event in event_id.items()}
        is_evaluation = session.startswith("evaluation")
        cue_codes = set(CLASS_EVENT_CODES)
        if is_evaluation:
            # The official evaluation GDF intentionally hides the class and
            # marks each cue as 783.  Its labels come from true_labels.zip.
            cue_codes.add("783")
        selected_events = [
            event for event in events if description_by_event.get(event[2]) in cue_codes
        ]
        if len(selected_events) != EXPECTED_TRIALS_PER_SESSION:
            raise OfficialDatasetError(
                f"{path.name} has {len(selected_events)} class cues; "
                f"expected {EXPECTED_TRIALS_PER_SESSION}"
            )
        selected_events_array = np.asarray(selected_events, dtype=int)
        event_to_label = {
            event: CLASS_EVENT_CODES[description]
            for event, description in description_by_event.items()
            if description in CLASS_EVENT_CODES
        }
        epoch_event_id = {
            f"event_{event}": event
            for event, description in description_by_event.items()
            if description in cue_codes
        }
        epochs = mne.Epochs(
            raw,
            selected_events_array,
            event_id=epoch_event_id,
            tmin=0.0,
            tmax=EPOCH_SECONDS,
            baseline=None,
            preload=True,
            picks=picks,
            event_repeated="drop",
            on_missing="raise",
            verbose="ERROR",
        )
        X = epochs.get_data(copy=True).astype(np.float32, copy=False)
        if is_evaluation:
            y = np.full(EXPECTED_TRIALS_PER_SESSION, -1, dtype=np.int64)
        else:
            y = np.asarray(
                [event_to_label[event[2]] for event in selected_events_array], dtype=np.int64
            )
        if X.shape != (EXPECTED_TRIALS_PER_SESSION, EXPECTED_CHANNELS, EXPECTED_SAMPLES):
            raise OfficialDatasetError(
                f"{path.name} produced shape {X.shape}; expected "
                f"({EXPECTED_TRIALS_PER_SESSION}, {EXPECTED_CHANNELS}, {EXPECTED_SAMPLES})"
            )
        if not is_evaluation:
            self._validate_label_balance(y, subject, session)
        return X, y

    def _trial_cache_key(self) -> str:
        """Return the cache key covering archives *and* the parsing code.

        ``provenance()["source_sha256"]`` alone is not enough: the band-pass,
        epoch start, baseline and session concatenation order are literals
        inside :meth:`_read_gdf`, and the descriptor restates them by hand.
        Editing the filter without editing the descriptor would leave the key
        unchanged and silently serve arrays built under the old preprocessing.
        Hashing this module's own source closes that gap -- any edit to the
        parsing path invalidates the cache whether or not the descriptor was
        kept in sync.
        """

        if self._trial_cache_key_cache is None:
            digest = hashlib.sha256()
            digest.update(str(self.provenance()["source_sha256"]).encode("utf-8"))
            digest.update(_sha256_file(Path(__file__)).encode("utf-8"))
            self._trial_cache_key_cache = digest.hexdigest()
        return self._trial_cache_key_cache

    def _trial_cache_path(self, subject: int) -> Path:
        """Return the on-disk trial cache path for one subject."""

        return (
            self.root
            / "trial_cache"
            / self._trial_cache_key()
            / f"S{subject:02d}.npz"
        )

    def _load_subject(self, subject: int) -> tuple[np.ndarray, np.ndarray]:
        if subject not in EXPECTED_SUBJECTS:
            raise ValueError(f"subject {subject} is not in official BCIC IV 2a")
        if subject in self._subject_cache:
            return self._subject_cache[subject]

        cache_path = self._trial_cache_path(subject)
        if cache_path.is_file():
            try:
                with np.load(cache_path) as bundle:
                    X, y = bundle["X"], bundle["y"]
            except (OSError, ValueError, KeyError):
                # A truncated or unreadable cache entry must never be fatal;
                # fall through and re-parse the official archive.
                cache_path.unlink(missing_ok=True)
            else:
                self._subject_cache[subject] = (X, y)
                return X, y

        X_train, y_train = self._read_gdf(self._gdf_path(subject, "T"), subject, "training")
        X_eval, y_from_events = self._read_gdf(self._gdf_path(subject, "E"), subject, "evaluation-events")
        y_eval = self._evaluation_labels(subject)
        if y_from_events.shape != y_eval.shape:
            raise OfficialDatasetError(
                f"evaluation cue count and official label count differ for subject {subject}"
            )
        X = np.concatenate((X_train, X_eval), axis=0)
        y = np.concatenate((y_train, y_eval), axis=0)
        if X.shape[0] != 2 * EXPECTED_TRIALS_PER_SESSION:
            raise OfficialDatasetError(f"subject {subject} did not yield 576 trials")
        if Counter(y.tolist()) != {label: 2 * EXPECTED_CLASS_TRIALS_PER_SESSION for label in range(EXPECTED_CLASSES)}:
            raise OfficialDatasetError(f"combined class balance is invalid for subject {subject}")
        self._subject_cache[subject] = (X, y)
        # Write through a temporary name so a crash mid-write cannot leave a
        # half-written archive that a later run would treat as valid.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # A PID is only unique within one node, and this cache directory is a
        # single shared /seu_share2 path, so two jobs on two nodes can draw the
        # same PID and open the same temporary file.  rename(2) is atomic only
        # for a unique source name, so let the OS pick the name.  The suffix
        # must be ``.npz`` because np.savez appends that extension when it is
        # absent, which would leave the renamed path empty.
        handle, tmp_name = tempfile.mkstemp(
            dir=str(cache_path.parent), prefix=f"{cache_path.stem}.", suffix=".npz"
        )
        os.close(handle)
        tmp_path = Path(tmp_name)
        try:
            with tmp_path.open("wb") as stream:
                np.savez(stream, X=X, y=y)
                stream.flush()
                # Publish only fully durable bytes: without the fsync a node
                # crash can leave a zero-filled entry that later reads as valid.
                os.fsync(stream.fileno())
            tmp_path.replace(cache_path)
        except OSError:
            tmp_path.unlink(missing_ok=True)
        return X, y

    def get_data(self, subjects: Iterable[int]) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        subject_ids = [int(subject) for subject in subjects]
        if not subject_ids:
            raise ValueError("at least one subject is required")
        arrays, labels = zip(*(self._load_subject(subject) for subject in subject_ids))
        return (
            np.concatenate(arrays, axis=0),
            np.concatenate(labels, axis=0),
            {"source": "official-gdf", "subjects": subject_ids},
        )

    def provenance(self) -> dict[str, object]:
        """Return a serializable, content-addressed data-source descriptor."""

        self._require_archive(self.gdf_archive, OFFICIAL_GDF_URL)
        self._require_archive(self.labels_archive, OFFICIAL_LABELS_URL)
        descriptor: dict[str, object] = {
            "kind": "official_bciciv_2a_gdf",
            "dataset_id": self.dataset_id,
            "source_urls": [OFFICIAL_GDF_URL, OFFICIAL_LABELS_URL],
            "archives": [
                {
                    "name": self.gdf_archive.name,
                    "bytes": self.gdf_archive.stat().st_size,
                    "sha256": self._archive_sha256(self.gdf_archive),
                },
                {
                    "name": self.labels_archive.name,
                    "bytes": self.labels_archive.stat().st_size,
                    "sha256": self._archive_sha256(self.labels_archive),
                },
            ],
            "subjects": list(EXPECTED_SUBJECTS),
            "preprocessing": {
                "eeg_channels": EXPECTED_CHANNELS,
                "eeg_channel_selection": {
                    "layout": list(OFFICIAL_GDF_CHANNEL_LAYOUT),
                    "included_eeg_channels": list(OFFICIAL_EEG_CHANNEL_NAMES),
                    "excluded_eog_channels": list(OFFICIAL_EOG_CHANNEL_NAMES),
                },
                "bandpass_hz": [8.0, 32.0],
                "epoch_reference": "class_cue",
                "epoch_seconds": EPOCH_SECONDS,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "samples_per_trial": EXPECTED_SAMPLES,
                "evaluation_labels": "official_true_labels_zip",
                "extraction_cache": {
                    "key": "archive_sha256",
                    "member_validation": "zip_size_and_crc32",
                },
            },
        }
        descriptor["source_sha256"] = _canonical_sha256(descriptor)
        return descriptor
