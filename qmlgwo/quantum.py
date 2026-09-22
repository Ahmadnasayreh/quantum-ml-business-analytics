"""
qmlgwo.quantum
==============
Variational and kernel quantum classifiers built on PennyLane state-vector
simulation.

Design
------
Every model below:

  1. builds an explicit parameterised circuit,
  2. measures a Pauli observable on the simulator,
  3. trains the circuit parameters against a differentiable loss using
     parameter-shift / adjoint gradients (Adam, SPSA) or a gradient-free
     method (COBYLA),
  4. seeds every stochastic step from :mod:`qmlgwo.config`.

All estimators follow the scikit-learn API (``get_params``/``set_params``/
``fit``/``predict``/``predict_proba``) so they compose inside ``Pipeline`` and
``cross_validate`` without leakage.

Complexity
----------
Let ``n`` = samples, ``d`` = features (= qubits), ``L`` = ansatz layers,
``E`` = epochs, ``P = L*d*3`` = trainable parameters.

  * ``QuantumVQC`` / ``QuantumDataReuploadingNN`` forward pass: O(n * L * d * 2^d);
    adjoint gradient adds a constant factor ~2, parameter-shift costs O(P)
    extra circuit evaluations. Training: O(E * n * L * d * 2^d).
  * ``QuantumKernelSVC``: O(n^2) circuit evaluations for the Gram matrix, each
    O(reps * d * 2^d)  ->  O(n^2 * reps * d * 2^d), plus O(n^3) for the SVM
    solve. This is why ``max_kernel_samples`` exists in the config.
  * ``QuantumEmbeddingTree``: O(n * d * 2^d) for the embedding + O(n log n * m)
    for the tree over ``m = 2d + (d-1)`` embedded features.

Space is O(2^d) for the state vector plus O(n * d) for the data.
"""

from __future__ import annotations

from typing import Callable, Literal, Sequence

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .config import get_rng

# --------------------------------------------------------------------------- #
# 1. Feature maps (data encoding)                                              #
# --------------------------------------------------------------------------- #


def zz_feature_map(x, n_qubits: int, reps: int = 2,
                   entanglement: Literal["linear", "circular", "full"] = "linear") -> None:
    """Second-order Pauli-Z evolution feature map (Havlicek et al., 2019).

    ``U(x) = [ exp(i * sum_j phi_j(x) Z_j + i * sum_{j<k} phi_jk(x) Z_j Z_k) H^{otimes n} ]^reps``

    with ``phi_j(x) = x_j`` and ``phi_jk(x) = (pi - x_j)(pi - x_k)``.

    The pairwise ``ZZ`` terms are what make this map classically hard to
    simulate in general; a product-state (RY-only) encoding, by contrast,
    factorises and admits a closed-form classical kernel.

    Supports PennyLane parameter broadcasting: ``x`` may be ``(d,)`` or ``(n, d)``.
    """
    pairs = _entangling_pairs(n_qubits, entanglement)
    for _ in range(reps):
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        for w in range(n_qubits):
            qml.RZ(2.0 * x[..., w], wires=w)
        for (i, j) in pairs:
            qml.CNOT(wires=[i, j])
            qml.RZ(2.0 * (np.pi - x[..., i]) * (np.pi - x[..., j]), wires=j)
            qml.CNOT(wires=[i, j])


def angle_feature_map(x, n_qubits: int, reps: int = 1, **_) -> None:
    """Plain RY angle encoding -- included only as an ablation baseline."""
    for _ in range(reps):
        for w in range(n_qubits):
            qml.RY(x[..., w], wires=w)


FEATURE_MAPS: dict[str, Callable] = {"zz": zz_feature_map, "angle": angle_feature_map}


def _entangling_pairs(n_qubits: int, kind: str) -> list[tuple[int, int]]:
    if n_qubits < 2:
        return []
    if kind == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    if kind == "circular":
        return [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    if kind == "full":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    raise ValueError(f"unknown entanglement pattern {kind!r}")


# --------------------------------------------------------------------------- #
# 2. Ansaetze (trainable blocks)                                               #
# --------------------------------------------------------------------------- #


def strongly_entangling_ansatz(weights, n_qubits: int) -> None:
    """``qml.StronglyEntanglingLayers``; weights shape ``(L, n_qubits, 3)``."""
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))


def ansatz_shape(n_layers: int, n_qubits: int) -> tuple[int, int, int]:
    return qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)


# --------------------------------------------------------------------------- #
# 3. Optimisers for circuit parameters                                         #
# --------------------------------------------------------------------------- #


def _train_adam(loss_fn, params, lr, n_epochs, batches, rng):
    """Vanilla Adam on parameter-shift/adjoint gradients."""
    m = pnp.zeros_like(params)
    v = pnp.zeros_like(params)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history, t = [], 0
    # No argnum/argnums keyword: renamed in PennyLane v0.42. Omitting it
    # differentiates w.r.t. the trainable arguments only -- X and y are passed
    # with requires_grad=False, so this yields d(loss)/d(params). Works on all versions.
    grad_fn = qml.grad(loss_fn)
    for _ in range(n_epochs):
        epoch_loss, n_b = 0.0, 0
        for Xb, yb in batches():
            t += 1
            n_b += 1
            g = grad_fn(params, Xb, yb)
            g = pnp.array(g, requires_grad=False)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g ** 2
            mhat = m / (1 - b1 ** t)
            vhat = v / (1 - b2 ** t)
            params = params - lr * mhat / (pnp.sqrt(vhat) + eps)
            epoch_loss += float(loss_fn(params, Xb, yb))
        # MEAN over batches, so the recorded value is on the same scale as a
        # full-batch loss and comparable across optimisers.
        history.append(epoch_loss / max(n_b, 1))
    return params, history


def _train_spsa(loss_fn, params, lr, n_epochs, batches, rng,
                c: float = 0.1, gamma: float = 0.101, alpha: float = 0.602):
    """Simultaneous Perturbation Stochastic Approximation (Spall, 1992).

    Two loss evaluations per step regardless of parameter count -- the standard
    gradient-free-ish baseline for noisy quantum hardware.
    """
    shape = params.shape
    flat = pnp.array(params.flatten(), requires_grad=False)
    history, k = [], 0
    for _ in range(n_epochs):
        epoch_loss, n_b = 0.0, 0
        for Xb, yb in batches():
            k += 1
            n_b += 1
            ak = lr / (k ** alpha)
            ck = c / (k ** gamma)
            delta = rng.choice([-1.0, 1.0], size=flat.shape)   # Rademacher
            lp = float(loss_fn(pnp.array((flat + ck * delta).reshape(shape),
                                         requires_grad=False), Xb, yb))
            lm = float(loss_fn(pnp.array((flat - ck * delta).reshape(shape),
                                         requires_grad=False), Xb, yb))
            ghat = (lp - lm) / (2.0 * ck) * (1.0 / delta)
            flat = flat - ak * ghat
            epoch_loss += 0.5 * (lp + lm)
        history.append(epoch_loss / max(n_b, 1))   # mean, see _train_adam
    return pnp.array(flat.reshape(shape), requires_grad=True), history


def _train_cobyla(loss_fn, params, lr, n_epochs, batches, rng, maxiter: int | None = None):
    """Gradient-free COBYLA on the full-batch loss (Powell, 1994)."""
    Xb, yb = next(iter(batches(full=True)))
    shape = params.shape
    history: list[float] = []

    def f(flat):
        val = float(loss_fn(pnp.array(flat.reshape(shape), requires_grad=False), Xb, yb))
        history.append(val)
        return val

    res = minimize(f, np.asarray(params).flatten(), method="COBYLA",
                   options={"maxiter": maxiter or (n_epochs * 20), "rhobeg": 0.5})
    return pnp.array(res.x.reshape(shape), requires_grad=True), history


CIRCUIT_OPTIMIZERS = {"adam": _train_adam, "spsa": _train_spsa, "cobyla": _train_cobyla}


# --------------------------------------------------------------------------- #
# 4. Base class                                                                #
# --------------------------------------------------------------------------- #


class _BaseVariationalClassifier(BaseEstimator, ClassifierMixin):
    """Shared training machinery for variational quantum classifiers.

    Parameters
    ----------
    n_layers : int
        Depth of the trainable ansatz (``reps`` in the manuscript).
    learning_rate, n_epochs, batch_size : float, int, int
        Standard optimisation controls.
    optimizer : {'adam', 'spsa', 'cobyla'}
        Circuit-parameter training algorithm.  Exposed so that gradient-based and
        gradient-free training can be compared directly.
    feature_map : {'zz', 'angle'}
    map_reps : int
        Repetitions of the data-encoding block.
    entanglement : {'linear', 'circular', 'full'}
    init_scale : float
        Std. dev. of the Gaussian parameter initialisation.
    diff_method : str
        ``'adjoint'`` (exact, fast) or ``'parameter-shift'`` (hardware-faithful).
    max_qubits : int
        Hard cap on simulated qubits.  Excess features are truncated and a
        warning attribute ``n_features_truncated_`` is set.
    random_state : int
    """

    _ansatz_uses_reuploading = False

    def __init__(self, n_layers: int = 2, learning_rate: float = 0.08,
                 n_epochs: int = 30, batch_size: int = 32,
                 optimizer: str = "adam", feature_map: str = "zz",
                 map_reps: int = 2, entanglement: str = "linear",
                 init_scale: float = 0.1, diff_method: str = "backprop",
                 max_qubits: int = 6, random_state: int = 0):
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.feature_map = feature_map
        self.map_reps = map_reps
        self.entanglement = entanglement
        self.init_scale = init_scale
        self.diff_method = diff_method
        self.max_qubits = max_qubits
        self.random_state = random_state

    # -- circuit construction ---------------------------------------------- #
    def _build_qnode(self, n_qubits: int):
        dev = qml.device("default.qubit", wires=n_qubits)
        fmap = FEATURE_MAPS[self.feature_map]
        reps, ent, reup = self.map_reps, self.entanglement, self._ansatz_uses_reuploading

        @qml.qnode(dev, diff_method=self.diff_method)
        def circuit(weights, x):
            if reup:
                # Data re-uploading (Perez-Salinas et al., 2020): the encoding is
                # interleaved with every trainable layer.  This is what keeps a
                # 1-qubit circuit -- the LDA case, where n_components = C-1 = 1 --
                # expressive enough to be a non-trivial classifier.
                for layer in range(weights.shape[0]):
                    fmap(x, n_qubits, reps=1, entanglement=ent)
                    qml.StronglyEntanglingLayers(weights[layer:layer + 1],
                                                 wires=range(n_qubits))
            else:
                fmap(x, n_qubits, reps=reps, entanglement=ent)
                strongly_entangling_ansatz(weights, n_qubits)
            return qml.expval(qml.PauliZ(0))

        return circuit

    # -- loss --------------------------------------------------------------- #
    def _make_loss(self, circuit, shape: tuple[int, ...], n_ansatz: int):
        """Binary cross-entropy on a trainable affine readout of ``<Z_0>``.

        ``logit = a * <Z_0>(theta; x) + b`` with ``(a, b)`` trained jointly with
        the circuit parameters.  Without this readout the decision threshold is
        pinned to ``<Z_0> = 0``, which is the main reason a plain expectation
        value underperforms on class-imbalanced data.

        Implemented in the log-sum-exp ("BCE-with-logits") form so that large
        ``|logit|`` cannot produce ``log(0)``.
        """
        def loss(params, X, y):
            weights = pnp.reshape(params[:n_ansatz], shape)
            a, b = params[n_ansatz], params[n_ansatz + 1]
            logit = a * circuit(weights, X) + b
            # log(1 + exp(-|z|)) + max(z, 0) - z * y   (stable BCE)
            return pnp.mean(pnp.log(1.0 + pnp.exp(-pnp.abs(logit)))
                            + pnp.maximum(logit, 0.0) - logit * y)
        return loss

    # -- sklearn API -------------------------------------------------------- #
    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64)
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("only binary classification is supported")
        y01 = (y == self.classes_[1]).astype(np.float64)

        self.n_features_in_ = X.shape[1]
        self.n_qubits_ = int(min(max(self.n_features_in_, 1), self.max_qubits))
        self.n_features_truncated_ = self.n_features_in_ > self.n_qubits_
        Xq = X[:, : self.n_qubits_]

        rng = get_rng("circuit_init", abs(int(self.random_state)) % 512)
        circuit = self._build_qnode(self.n_qubits_)
        self._circuit = circuit

        shape = ansatz_shape(int(self.n_layers), self.n_qubits_)
        n_ansatz = int(np.prod(shape))
        self._shape, self._n_ansatz = shape, n_ansatz
        self.n_parameters_ = n_ansatz + 2          # + readout scale and bias
        loss_fn = self._make_loss(circuit, shape, n_ansatz)

        # theta ~ N(0, init_scale^2); readout initialised at (a, b) = (1, 0).
        init = np.concatenate([rng.normal(0.0, self.init_scale, n_ansatz), [1.0, 0.0]])
        weights = pnp.array(init, requires_grad=True)

        Xp = pnp.array(Xq, requires_grad=False)
        yp = pnp.array(y01, requires_grad=False)
        bs = int(max(8, min(self.batch_size, len(Xq))))

        def batches(full: bool = False):
            if full:
                yield Xp, yp
                return
            idx = rng.permutation(len(Xq))
            for s in range(0, len(idx), bs):
                sel = idx[s:s + bs]
                if len(sel) < 2:
                    continue
                yield Xp[sel], yp[sel]

        trainer = CIRCUIT_OPTIMIZERS[self.optimizer]
        weights, history = trainer(loss_fn, weights, self.learning_rate,
                                   int(self.n_epochs), batches, rng)

        self.params_ = np.asarray(weights, dtype=np.float64)
        self.weights_ = self.params_[:n_ansatz].reshape(shape)
        self.readout_ = tuple(self.params_[n_ansatz:])          # (a, b)
        self.loss_history_ = [float(v) for v in history]
        self.is_fitted_ = True
        return self

    def decision_function(self, X):
        """Return the pre-sigmoid logit ``a * <Z_0> + b``."""
        check_is_fitted(self, "is_fitted_")
        X = check_array(X, dtype=np.float64)
        Xq = pnp.array(X[:, : self.n_qubits_], requires_grad=False)
        w = pnp.array(self.weights_, requires_grad=False)
        e = np.asarray(self._circuit(w, Xq), dtype=np.float64).ravel()
        a, b = self.readout_
        return a * e + b

    def predict_proba(self, X):
        z = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    def __sklearn_is_fitted__(self) -> bool:
        return getattr(self, "is_fitted_", False)


# --------------------------------------------------------------------------- #
# 5. Concrete models                                                           #
# --------------------------------------------------------------------------- #


class QuantumVQC(_BaseVariationalClassifier):
    """Variational Quantum Classifier: ZZ feature map + strongly-entangling ansatz.

    Encoding block applied once (``map_reps`` repetitions), followed by
    ``n_layers`` trainable layers; readout is ``<Z_0>``.
    """
    _ansatz_uses_reuploading = False


class QuantumReuploadingNN(_BaseVariationalClassifier):
    """Quantum Neural Network with data re-uploading.

    Encoding and trainable layers alternate, so circuit depth in the data grows
    with ``n_layers``.  This is a genuinely different architecture from
    :class:`QuantumVQC` (rather than a renamed copy) and remains expressive at
    ``n_qubits = 1``, which is the regime forced by LDA on a binary task.
    """
    _ansatz_uses_reuploading = True

    def __init__(self, n_layers: int = 3, learning_rate: float = 0.08,
                 n_epochs: int = 30, batch_size: int = 32, optimizer: str = "adam",
                 feature_map: str = "angle", map_reps: int = 1,
                 entanglement: str = "linear", init_scale: float = 0.1,
                 diff_method: str = "backprop", max_qubits: int = 6,
                 random_state: int = 0):
        super().__init__(n_layers=n_layers, learning_rate=learning_rate,
                         n_epochs=n_epochs, batch_size=batch_size,
                         optimizer=optimizer, feature_map=feature_map,
                         map_reps=map_reps, entanglement=entanglement,
                         init_scale=init_scale, diff_method=diff_method,
                         max_qubits=max_qubits, random_state=random_state)


class QuantumKernelSVC(BaseEstimator, ClassifierMixin):
    """Support vector classifier on a fidelity quantum kernel.

    ``K(x, x') = |<0| U(x')^dagger U(x) |0>|^2`` evaluated by the
    compute-uncompute circuit.  With the entangling ZZ map this kernel has no
    known efficient classical closed form -- unlike the product-state RY
    encoding, whose kernel factorises as ``prod_j cos^2((x_j - x'_j)/2)``.

    Notes
    -----
    Gram-matrix construction costs O(n^2) circuit evaluations.  ``max_samples``
    caps the training set via a *seeded, stratified* subsample.
    """

    def __init__(self, C: float = 1.0, map_reps: int = 2,
                 entanglement: str = "linear", scale: float = 1.0,
                 max_samples: int = 600, max_qubits: int = 6,
                 random_state: int = 0):
        self.C = C
        self.map_reps = map_reps
        self.entanglement = entanglement
        self.scale = scale
        self.max_samples = max_samples
        self.max_qubits = max_qubits
        self.random_state = random_state

    def _build_kernel_qnode(self, n_qubits: int):
        dev = qml.device("default.qubit", wires=n_qubits)
        reps, ent = self.map_reps, self.entanglement

        @qml.qnode(dev)
        def kernel_circuit(x1, x2):
            zz_feature_map(x1, n_qubits, reps=reps, entanglement=ent)
            qml.adjoint(zz_feature_map)(x2, n_qubits, reps=reps, entanglement=ent)
            return qml.probs(wires=range(n_qubits))

        return kernel_circuit

    def _gram(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Vectorised Gram matrix via PennyLane parameter broadcasting."""
        nA, nB = len(A), len(B)
        K = np.empty((nA, nB), dtype=np.float64)
        for i in range(nA):
            x1 = np.repeat(A[i][None, :], nB, axis=0)
            probs = self._kernel_circuit(x1, B)
            K[i] = np.asarray(probs)[:, 0]      # |<0...0|psi>|^2
        return K

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        self.n_qubits_ = int(min(max(self.n_features_in_, 1), self.max_qubits))

        rng = get_rng("resampling", abs(int(self.random_state)) % 512)
        if len(X) > self.max_samples:
            # Stratified, seeded subsample -- never an unseeded np.random.choice.
            idx = np.concatenate([
                rng.choice(np.flatnonzero(y == c),
                           size=max(1, int(round(self.max_samples * np.mean(y == c)))),
                           replace=False)
                for c in self.classes_
            ])
            rng.shuffle(idx)
            X, y = X[idx], y[idx]

        self.X_fit_ = X[:, : self.n_qubits_] * self.scale
        self.y_fit_ = y
        self._kernel_circuit = self._build_kernel_qnode(self.n_qubits_)

        K = self._gram(self.X_fit_, self.X_fit_)
        K = 0.5 * (K + K.T)                      # enforce exact symmetry
        self.svc_ = SVC(C=self.C, kernel="precomputed", probability=True,
                        random_state=abs(int(self.random_state)) % (2**31 - 1))
        self.svc_.fit(K, y)
        self.is_fitted_ = True
        return self

    def _transform(self, X):
        X = check_array(X, dtype=np.float64)
        return self._gram(X[:, : self.n_qubits_] * self.scale, self.X_fit_)

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        return self.svc_.predict(self._transform(X))

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        return self.svc_.predict_proba(self._transform(X))

    def __sklearn_is_fitted__(self) -> bool:
        return getattr(self, "is_fitted_", False)


class QuantumEmbeddingTree(BaseEstimator, ClassifierMixin):
    """Decision tree over a quantum feature embedding.

    Honest naming matters here.  A decision tree is not a quantum algorithm;
    what is quantum is the *representation* it splits on.  Each sample is mapped
    through the ZZ feature map and summarised by the expectation values
    ``{<Z_j>}, {<X_j>}, {<Z_j Z_{j+1}>}`` (``3d - 1`` features for ``d`` qubits).
    A CART tree is then fitted on that embedding.

    The manuscript must describe this model in exactly these terms rather than
    as a "quantum decision tree" with quantum splitting criteria.
    """

    def __init__(self, max_depth: int = 4, min_samples_split: int = 8,
                 map_reps: int = 2, entanglement: str = "linear",
                 max_qubits: int = 6, random_state: int = 0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.map_reps = map_reps
        self.entanglement = entanglement
        self.max_qubits = max_qubits
        self.random_state = random_state

    def _build_embed_qnode(self, n_qubits: int):
        dev = qml.device("default.qubit", wires=n_qubits)
        reps, ent = self.map_reps, self.entanglement
        pairs = _entangling_pairs(n_qubits, "linear")

        @qml.qnode(dev)
        def embed(x):
            zz_feature_map(x, n_qubits, reps=reps, entanglement=ent)
            obs = [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]
            obs += [qml.expval(qml.PauliX(w)) for w in range(n_qubits)]
            obs += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for (i, j) in pairs]
            return obs

        return embed

    def _embed(self, X: np.ndarray) -> np.ndarray:
        out = self._embed_circuit(X[:, : self.n_qubits_])
        return np.asarray(out, dtype=np.float64).T   # (n_samples, n_observables)

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        self.n_qubits_ = int(min(max(self.n_features_in_, 1), self.max_qubits))
        self._embed_circuit = self._build_embed_qnode(self.n_qubits_)

        Z = self._embed(X)
        self.embedding_dim_ = Z.shape[1]
        self.tree_ = DecisionTreeClassifier(
            max_depth=int(self.max_depth),
            min_samples_split=int(self.min_samples_split),
            random_state=abs(int(self.random_state)) % (2**31 - 1),
        ).fit(Z, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        return self.tree_.predict(self._embed(check_array(X, dtype=np.float64)))

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        return self.tree_.predict_proba(self._embed(check_array(X, dtype=np.float64)))

    def __sklearn_is_fitted__(self) -> bool:
        return getattr(self, "is_fitted_", False)


__all__ = [
    "zz_feature_map", "angle_feature_map", "FEATURE_MAPS", "ansatz_shape",
    "QuantumVQC", "QuantumReuploadingNN", "QuantumKernelSVC",
    "QuantumEmbeddingTree", "CIRCUIT_OPTIMIZERS",
]
