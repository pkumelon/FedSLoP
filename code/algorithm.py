import numpy as np
from typing import Callable, Tuple, List, Dict, Any, Optional
import random


class FedSLopAlgorithm:
    """FedSLop：客户端低秩动量 + 全维通信 + 服务器端动量。

    设计目标：
    - 通信与 FedAvg-M 相同：
      * 下行：服务器广播全局参数 theta ∈ R^d；
      * 上行：客户端上传全维本地模型 theta_k ∈ R^d（或其差分 Δθ_k）。
    - 内存节省：
      * 客户端只在低维子空间中维护动量 m_k^r ∈ R^r（r << d），而不是全维动量；
      * 服务器端维护一个全局动量向量 v_server ∈ R^d。
    - 子空间基 P_t：
      * 每轮重采样一个 P_t ∈ R^{d×r}，通过随机种子控制，使得"P_t 的生成过程"可复现；
      * 客户端不显式存储 P_t，只需保存其低维动量 m_k^r，下一轮在新的 P_{t+1} 下继续。

    注意：
    - 本实现中，P_t 在服务器端每轮重新生成，并在客户端端"按相同随机种子策略"生成；
      在当前纯函数实现里，我们直接在 optimize 内共享同一个 P_t（等价于通过种子同步）。
    - 通信量与 FedAvgMOptimizer 相同（全维模型上传），仅在客户端动量维度上节省内存。
    """

    def __init__(
        self,
        rounds: int = 100,
        local_steps: int = 1,
        client_fraction: float = 1.0,
        stepsize: float = 0.01,
        momentum: float = 0.9,
        batch_size: int = 32,
        proj_rank: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.rounds = rounds
        self.local_steps = local_steps
        self.client_fraction = client_fraction
        self.stepsize = stepsize
        self.momentum = momentum
        self.batch_size = batch_size
        self.proj_rank = proj_rank
        self.random_state = random_state
        self.history: List[float] = []
        # 存储的是低维动量 m_k^r ∈ R^{r}
        self._client_momentum: Dict[int, np.ndarray] = {}
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def _linear_regression_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """线性回归梯度：1/|B| X^T (Xw - y)。"""
        preds = X @ w
        diff = preds - y
        grad = X.T @ diff / X.shape[0]
        return grad

    def _sample_minibatch(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        if batch_size >= n:
            return X, y
        idx = np.random.choice(n, size=batch_size, replace=False)
        return X[idx], y[idx]

    def _build_subspace_basis(self, dim: int, rank: Optional[int]) -> np.ndarray:
        """构造低秩子空间基矩阵 P ∈ R^{d×r}。

        - 若 rank 为 None 或 ≥ dim，则退化为全维子空间，P = I_d；
        - 否则使用高斯矩阵 + QR 得到正交基，取前 r 列。
        """
        if rank is None or rank >= dim:
            return np.eye(dim)
        r = rank
        G = np.random.randn(dim, r)
        Q, _ = np.linalg.qr(G)
        P = Q[:, :r]
        return P

    def optimize(
        self,
        objective_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        x0: np.ndarray,
        client_datasets: List[Tuple[np.ndarray, np.ndarray]],
        max_iter: Optional[int] = None,
        tol: float = 1e-6,
    ) -> Tuple[np.ndarray, float]:
        """运行 FedSLop（客户端低秩动量 + 全维通信 + 服务器端动量）。

        通信协议：
        - 下行：服务器广播全局参数 theta ∈ R^d；
        - 上行：客户端上传全维本地模型 theta_k ∈ R^d；
        - 服务器对本地模型求平均，得到 theta_mean，并在服务器端维护动量 v_server。

        客户端动量：
        - 每个客户端 k 维护低维动量 m_k^r ∈ R^r；
        - 每轮使用当前的子空间基 P_t，将梯度投影到子空间并更新 m_k^r；
        - 再通过 P_t 将 m_k^r 映射回全维空间进行参数更新。

        服务器端动量：
        - v_server ← μ v_server + (theta_mean − theta)
        - theta ← theta + v_server

        Args:
            objective_func: 目标函数 f(X, y, w) -> float。
            x0: 初始全局参数向量。
            client_datasets: 客户端数据列表 [(X_k, y_k), ...]。
            max_iter: 最大迭代轮数（若为 None，则使用 self.rounds）。
            tol: 容忍度，用于简单早停。

        Returns:
            (最终参数, 最终目标值)。
        """
        theta = x0.copy()
        v_server = np.zeros_like(theta)  # 服务器端动量
        num_clients = len(client_datasets)
        rounds = max_iter if max_iter is not None else self.rounds
        self.history.clear()

        for t in range(rounds):
            dim = theta.shape[0]
            # 每轮重采样子空间基 P_t，通过随机种子保证"生成过程可复现"
            if self.random_state is not None:
                # 使用 (random_state + t) 作为本轮的种子，模拟"通过种子传输 P_t"
                np.random.seed(self.random_state + t)
            P_t = self._build_subspace_basis(dim, self.proj_rank)
            r = P_t.shape[1]

            m = max(1, int(self.client_fraction * num_clients))
            selected = np.random.choice(num_clients, size=m, replace=False)
            local_thetas: List[np.ndarray] = []

            for idx in selected:
                X_k, y_k = client_datasets[idx]
                theta_local = theta.copy()

                # 取出客户端的低维动量 m_k^r；若不存在则初始化为 0
                m_prev = self._client_momentum.get(idx)
                if m_prev is None or m_prev.shape[0] != r:
                    m_local = np.zeros(r, dtype=theta.dtype)
                else:
                    m_local = m_prev.copy()

                for _ in range(self.local_steps):
                    X_b, y_b = self._sample_minibatch(X_k, y_k, self.batch_size)
                    g = self._linear_regression_grad(X_b, y_b, theta_local)  # R^{d}
                    # 梯度投影到低维：g_r = P_t^T g ∈ R^{r}
                    g_r = P_t.T @ g
                    # 在低维空间中更新客户端动量
                    m_local = self.momentum * m_local + g_r
                    # 将低维动量映射回原空间：v_full = P_t m_local ∈ R^{d}
                    v_full = P_t @ m_local
                    theta_local = theta_local - self.stepsize * v_full

                # 客户端上传全维本地模型（或等价的全维差分）
                local_thetas.append(theta_local)
                # 存储低维动量 m_k^r，跨轮次复用
                self._client_momentum[idx] = m_local.copy()

            if not local_thetas:
                break

            # 服务器端：对本地模型求平均，然后做服务器动量更新
            theta_mean = np.mean(local_thetas, axis=0)
            delta = theta_mean - theta
            v_server = self.momentum * v_server + delta
            theta = theta + v_server

            # 计算全局损失
            losses = []
            for X_k, y_k in client_datasets:
                losses.append(objective_func(X_k, y_k, theta))
            mean_loss = float(np.mean(losses))
            self.history.append(mean_loss)

            if t > 0 and abs(self.history[-2] - self.history[-1]) < tol:
                break

        final_loss = self.history[-1] if self.history else objective_func(
            client_datasets[0][0], client_datasets[0][1], theta
        )
        return theta, final_loss

    def get_convergence_history(self) -> List[float]:
        return self.history


class FedMefAlgorithm:
    """简化版 FedMef 稀疏联邦训练算法（单层向量掩码）。"""

    def __init__(
        self,
        rounds: int = 100,
        local_epochs: int = 1,
        client_fraction: float = 1.0,
        base_lr: float = 0.01,
        adjust_interval_rounds: int = 10,
        stop_adjust_round: int = 100,
        grow_ratio: float = 0.01,
        random_state: Optional[int] = None,
    ) -> None:
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.client_fraction = client_fraction
        self.base_lr = base_lr
        self.adjust_interval_rounds = adjust_interval_rounds
        self.stop_adjust_round = stop_adjust_round
        self.grow_ratio = grow_ratio
        self.history: List[float] = []
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def _linear_regression_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        preds = X @ w
        diff = preds - y
        grad = X.T @ diff / X.shape[0]
        return grad

    def _sample_minibatch(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        if batch_size >= n:
            return X, y
        idx = np.random.choice(n, size=batch_size, replace=False)
        return X[idx], y[idx]

    def _adjust_mask(self, theta: np.ndarray, mask: np.ndarray, grad_est: np.ndarray) -> np.ndarray:
        """根据梯度估计与权重幅度调整掩码。"""
        d = theta.shape[0]
        k = max(1, int(self.grow_ratio * d))

        # 生长：在被剪枝位置中选择梯度绝对值最大的 k 个
        pruned_idx = np.where(mask == 0)[0]
        if pruned_idx.size > 0:
            pruned_grad = np.abs(grad_est[pruned_idx])
            grow_k = min(k, pruned_idx.size)
            grow_indices = pruned_idx[np.argpartition(-pruned_grad, grow_k - 1)[:grow_k]]
        else:
            grow_indices = np.array([], dtype=int)

        # 剪枝：在激活位置中选择权重幅度最小的 k 个
        active_idx = np.where(mask == 1)[0]
        if active_idx.size > 0:
            active_abs = np.abs(theta[active_idx])
            drop_k = min(k, active_idx.size)
            drop_indices = active_idx[np.argpartition(active_abs, drop_k - 1)[:drop_k]]
        else:
            drop_indices = np.array([], dtype=int)

        new_mask = mask.copy()
        new_mask[grow_indices] = 1
        new_mask[drop_indices] = 0
        return new_mask

    def optimize(
        self,
        objective_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        x0: np.ndarray,
        client_datasets: List[Tuple[np.ndarray, np.ndarray]],
        max_iter: Optional[int] = None,
        tol: float = 1e-6,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, float]:
        """运行简化版 FedMef 联邦稀疏训练。

        Args:
            objective_func: 目标函数 f(X, y, w) -> float。
            x0: 初始参数向量。
            client_datasets: 客户端数据列表。
            max_iter: 最大轮数（若为 None，则使用 self.rounds）。
            tol: 容忍度。
            batch_size: 本地小批量大小。
        """
        theta = x0.copy()
        d = theta.shape[0]
        mask = np.ones_like(theta)
        num_clients = len(client_datasets)
        rounds = max_iter if max_iter is not None else self.rounds
        self.history.clear()

        for r in range(rounds):
            m = max(1, int(self.client_fraction * num_clients))
            selected = np.random.choice(num_clients, size=m, replace=False)
            local_thetas = []

            for idx in selected:
                X_k, y_k = client_datasets[idx]
                theta_k = theta.copy()
                for _ in range(self.local_epochs):
                    X_b, y_b = self._sample_minibatch(X_k, y_k, batch_size)
                    g = self._linear_regression_grad(X_b, y_b, theta_k)
                    theta_k = theta_k - self.base_lr * g * mask
                local_thetas.append(theta_k)

            if not local_thetas:
                break
            theta = np.mean(local_thetas, axis=0)

            # 掩码调整
            if (r % self.adjust_interval_rounds == 0) and (r <= self.stop_adjust_round):
                # 简单有限差分估计全局梯度方向
                eps = 1e-3
                loss_plus = objective_func(client_datasets[0][0], client_datasets[0][1], theta + eps)
                loss_minus = objective_func(client_datasets[0][0], client_datasets[0][1], theta - eps)
                grad_est = (loss_plus - loss_minus) / (2 * eps)
                # 这里使用标量估计，实际应为向量梯度；为保持结构，使用 theta 的符号近似
                grad_vec = np.sign(theta) * grad_est
                mask = self._adjust_mask(theta, mask, grad_vec)
                theta = theta * mask

            # 记录损失
            losses = [objective_func(X_k, y_k, theta) for X_k, y_k in client_datasets]
            mean_loss = float(np.mean(losses))
            self.history.append(mean_loss)
            if r > 0 and abs(self.history[-2] - self.history[-1]) < tol:
                break

        final_loss = self.history[-1] if self.history else objective_func(
            client_datasets[0][0], client_datasets[0][1], theta
        )
        return theta, final_loss

    def get_convergence_history(self) -> List[float]:
        return self.history


class FederatedSelectAlgorithm:
    """抽象的 Federated Select 联邦训练框架。

    具体的选择函数 psi、客户端更新与聚合逻辑由用户注入。
    """

    def __init__(
        self,
        rounds: int = 100,
        client_fraction: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        self.rounds = rounds
        self.client_fraction = client_fraction
        self.history: List[float] = []
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def optimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        x0: np.ndarray,
        num_clients: int,
        select_keys_fn: Callable[[int, np.ndarray], Any],
        psi: Callable[[np.ndarray, Any], Any],
        client_update: Callable[[Any, int], Any],
        aggregate: Callable[[List[Any], List[Any]], Any],
        server_update: Callable[[np.ndarray, Any], np.ndarray],
        max_iter: Optional[int] = None,
        tol: float = 1e-6,
    ) -> Tuple[np.ndarray, float]:
        """运行 Federated Select 训练。

        Args:
            objective_func: 仅依赖服务器状态 x 的目标函数。
            x0: 初始服务器状态向量。
            num_clients: 客户端总数。
            select_keys_fn: 生成客户端选择键 z_n^t 的函数。
            psi: 服务器侧选择函数 psi(x_t, z_n^t)。
            client_update: 客户端更新函数 ClientUpdate(y_n^t, client_id)。
            aggregate: 聚合函数 Aggregate({u_n^t}, {z_n^t})。
            server_update: 服务器更新函数 ServerUpdate(x_t, u^t)。
        """
        x = x0.copy()
        rounds = max_iter if max_iter is not None else self.rounds
        self.history.clear()

        for t in range(rounds):
            m = max(1, int(self.client_fraction * num_clients))
            selected = np.random.choice(num_clients, size=m, replace=False)
            updates: List[Any] = []
            keys: List[Any] = []

            for cid in selected:
                z = select_keys_fn(cid, x)
                y = psi(x, z)
                u = client_update(y, cid)
                updates.append(u)
                keys.append(z)

            if not updates:
                break

            u_global = aggregate(updates, keys)
            x = server_update(x, u_global)

            loss = float(objective_func(x))
            self.history.append(loss)
            if t > 0 and abs(self.history[-2] - self.history[-1]) < tol:
                break

        final_loss = self.history[-1] if self.history else float(objective_func(x))
        return x, final_loss

    def get_convergence_history(self) -> List[float]:
        return self.history


class NeuLiteAlgorithm:
    """简化版 NeuLite 参数协同适应训练范式。

    将参数向量划分为若干块，每轮仅更新一个块和最后的 Op 块。
    """

    def __init__(
        self,
        rounds: int = 100,
        num_blocks: int = 4,
        client_fraction: float = 1.0,
        local_epochs: int = 1,
        stepsize: float = 0.01,
        random_state: Optional[int] = None,
    ) -> None:
        self.rounds = rounds
        self.num_blocks = num_blocks
        self.client_fraction = client_fraction
        self.local_epochs = local_epochs
        self.stepsize = stepsize
        self.history: List[float] = []
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def _split_blocks(self, theta: np.ndarray) -> List[np.ndarray]:
        d = theta.shape[0]
        sizes = [d // self.num_blocks] * self.num_blocks
        for i in range(d % self.num_blocks):
            sizes[i] += 1
        blocks = []
        start = 0
        for sz in sizes:
            blocks.append(theta[start : start + sz])
            start += sz
        return blocks

    def _merge_blocks(self, blocks: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(blocks, axis=0)

    def optimize(
        self,
        objective_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        x0: np.ndarray,
        client_datasets: List[Tuple[np.ndarray, np.ndarray]],
        grad_func: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
        max_iter: Optional[int] = None,
        tol: float = 1e-6,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, float]:
        """运行简化版 NeuLite 联邦训练。

        Args:
            objective_func: 目标函数 f(X, y, w) -> float。
            x0: 初始参数向量。
            client_datasets: 客户端数据列表。
            grad_func: 梯度函数，如果为 None，则使用线性回归梯度。
        """
        if grad_func is None:
            def grad_func(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:  # type: ignore[no-redef]
                preds = X @ w
                diff = preds - y
                return X.T @ diff / X.shape[0]

        theta = x0.copy()
        num_clients = len(client_datasets)
        rounds = max_iter if max_iter is not None else self.rounds
        self.history.clear()

        for r in range(rounds):
            blocks = self._split_blocks(theta)
            T_blocks = len(blocks)
            op_idx = T_blocks - 1
            t_idx = r % T_blocks

            m = max(1, int(self.client_fraction * num_clients))
            selected = np.random.choice(num_clients, size=m, replace=False)
            agg_blocks = [np.zeros_like(b) for b in blocks]

            for idx in selected:
                X_k, y_k = client_datasets[idx]
                local_blocks = [b.copy() for b in blocks]
                for _ in range(self.local_epochs):
                    theta_local = self._merge_blocks(local_blocks)
                    n = X_k.shape[0]
                    if batch_size >= n:
                        X_b, y_b = X_k, y_k
                    else:
                        bid = np.random.choice(n, size=batch_size, replace=False)
                        X_b, y_b = X_k[bid], y_k[bid]
                    g = grad_func(X_b, y_b, theta_local)
                    # 拆分梯度为块
                    g_blocks = self._split_blocks(g)
                    # 仅更新当前块和 Op 块
                    local_blocks[t_idx] = local_blocks[t_idx] - self.stepsize * g_blocks[t_idx]
                    if op_idx != t_idx:
                        local_blocks[op_idx] = local_blocks[op_idx] - self.stepsize * g_blocks[op_idx]

                for j, b in enumerate(local_blocks):
                    agg_blocks[j] += b

            if selected.size > 0:
                for j in range(len(agg_blocks)):
                    agg_blocks[j] /= selected.size
                theta = self._merge_blocks(agg_blocks)

            losses = [objective_func(X_k, y_k, theta) for X_k, y_k in client_datasets]
            mean_loss = float(np.mean(losses))
            self.history.append(mean_loss)
            if r > 0 and abs(self.history[-2] - self.history[-1]) < tol:
                break

        final_loss = self.history[-1] if self.history else objective_func(
            client_datasets[0][0], client_datasets[0][1], theta
        )
        return theta, final_loss

    def get_convergence_history(self) -> List[float]:
        return self.history


class FedAvgMOptimizer:
    """带服务器动量的 FedAvg 优化器。"""

    def __init__(
        self,
        rounds: int = 100,
        local_epochs: int = 1,
        client_fraction: float = 1.0,
        stepsize: float = 1.0,
        momentum: float = 0.9,
        random_state: Optional[int] = None,
    ) -> None:
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.client_fraction = client_fraction
        self.stepsize = stepsize
        self.momentum = momentum
        self.history: List[float] = []
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def _linear_regression_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        preds = X @ w
        diff = preds - y
        grad = X.T @ diff / X.shape[0]
        return grad

    def optimize(
        self,
        objective_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        x0: np.ndarray,
        client_datasets: List[Tuple[np.ndarray, np.ndarray]],
        max_iter: Optional[int] = None,
        tol: float = 1e-6,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, float]:
        """运行带动量的 FedAvg。

        Args:
            objective_func: 目标函数 f(X, y, w) -> float。
            x0: 初始参数向量。
            client_datasets: 客户端数据列表。
        """
        theta = x0.copy()
        v = np.zeros_like(theta)
        num_clients = len(client_datasets)
        rounds = max_iter if max_iter is not None else self.rounds
        self.history.clear()

        for r in range(rounds):
            m = max(1, int(self.client_fraction * num_clients))
            selected = np.random.choice(num_clients, size=m, replace=False)
            local_thetas = []

            for idx in selected:
                X_k, y_k = client_datasets[idx]
                theta_k = theta.copy()
                for _ in range(self.local_epochs):
                    n = X_k.shape[0]
                    if batch_size >= n:
                        X_b, y_b = X_k, y_k
                    else:
                        bid = np.random.choice(n, size=batch_size, replace=False)
                        X_b, y_b = X_k[bid], y_k[bid]
                    g = self._linear_regression_grad(X_b, y_b, theta_k)
                    theta_k = theta_k - self.stepsize * g
                local_thetas.append(theta_k)

            if not local_thetas:
                break
            theta_mean = np.mean(local_thetas, axis=0)
            delta = theta_mean - theta
            v = self.momentum * v + delta
            theta = theta + v

            losses = [objective_func(X_k, y_k, theta) for X_k, y_k in client_datasets]
            mean_loss = float(np.mean(losses))
            self.history.append(mean_loss)
            if r > 0 and abs(self.history[-2] - self.history[-1]) < tol:
                break

        final_loss = self.history[-1] if self.history else objective_func(
            client_datasets[0][0], client_datasets[0][1], theta
        )
        return theta, final_loss

    def get_convergence_history(self) -> List[float]:
        return self.history


class FedLoRAMOptimizer:
    """简化版 FedLoRA 带动量优化器（适配向量参数场景）。

    将参数视为列向量 w ∈ R^{d×1}，表示为 base + A @ B，
    其中 A ∈ R^{d×r}, B ∈ R^{r×1}。客户端仅更新 A, B，
    服务器对 A, B 的变化量进行动量聚合。
    """

    def __init__(
        self,
        rounds: int = 100,
        local_epochs: int = 1,
        client_fraction: float = 1.0,
        stepsize: float = 1.0,
        momentum: float = 0.9,
        lora_rank: int = 4,
        random_state: Optional[int] = None,
    ) -> None:
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.client_fraction = client_fraction
        self.stepsize = stepsize
        self.momentum = momentum
        self.lora_rank = lora_rank
        self.history: List[float] = []
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def _linear_regression_grad(self, X: np.ndarray, y: np.ndarray, w_col: np.ndarray) -> np.ndarray:
        """线性回归梯度，w_col 视为列向量 (d,1)。"""
        # 展平为 (d,) 使用前向计算
        w = w_col.reshape(-1)
        preds = X @ w
        diff = preds - y
        grad = X.T @ diff / X.shape[0]
        # 返回列向量形状 (d,1)
        return grad.reshape(-1, 1)

    def optimize(
        self,
        objective_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        x0: np.ndarray,
        client_datasets: List[Tuple[np.ndarray, np.ndarray]],
        max_iter: Optional[int] = None,
        tol: float = 1e-6,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, float]:
        """运行简化版 FedLoRA 带动量训练。

        Args:
            objective_func: 目标函数 f(X, y, w) -> float，其中 w 为展平后的向量。
            x0: 初始基础参数向量（base）。
            client_datasets: 客户端数据列表。
        """
        # 将基础参数表示为列向量
        base_col = x0.reshape(-1, 1)
        d = base_col.shape[0]
        r = self.lora_rank
        # 初始化低秩因子和动量
        A = np.random.randn(d, r) * 0.01
        B = np.random.randn(r, 1) * 0.01
        v_A = np.zeros_like(A)
        v_B = np.zeros_like(B)

        num_clients = len(client_datasets)
        rounds = max_iter if max_iter is not None else self.rounds
        self.history.clear()

        for it in range(rounds):
            m = max(1, int(self.client_fraction * num_clients))
            selected = np.random.choice(num_clients, size=m, replace=False)
            sum_dA = np.zeros_like(A)
            sum_dB = np.zeros_like(B)

            for idx in selected:
                X_k, y_k = client_datasets[idx]
                A_k = A.copy()
                B_k = B.copy()
                for _ in range(self.local_epochs):
                    w_col = base_col + A_k @ B_k  # (d,1)
                    n = X_k.shape[0]
                    if batch_size >= n:
                        X_b, y_b = X_k, y_k
                    else:
                        bid = np.random.choice(n, size=batch_size, replace=False)
                        X_b, y_b = X_k[bid], y_k[bid]
                    g_w_col = self._linear_regression_grad(X_b, y_b, w_col)  # (d,1)
                    # 简化的梯度到 A, B 的分配：
                    # dL/dA ≈ g_w_col @ B_k.T, 形状 (d,r)
                    # dL/dB ≈ A_k.T @ g_w_col, 形状 (r,1)
                    g_A = g_w_col @ B_k.T / d
                    g_B = A_k.T @ g_w_col / d
                    A_k = A_k - self.stepsize * g_A
                    B_k = B_k - self.stepsize * g_B

                sum_dA += (A_k - A)
                sum_dB += (B_k - B)

            if selected.size > 0:
                mean_dA = sum_dA / selected.size
                mean_dB = sum_dB / selected.size
                v_A = self.momentum * v_A + mean_dA
                v_B = self.momentum * v_B + mean_dB
                A = A + v_A
                B = B + v_B

            w_col_global = base_col + A @ B
            w_global = w_col_global.reshape(-1)
            losses = [objective_func(X_k, y_k, w_global) for X_k, y_k in client_datasets]
            mean_loss = float(np.mean(losses))
            self.history.append(mean_loss)
            if it > 0 and abs(self.history[-2] - self.history[-1]) < tol:
                break

        w_col_final = base_col + A @ B
        w_final = w_col_final.reshape(-1)
        final_loss = self.history[-1] if self.history else objective_func(
            client_datasets[0][0], client_datasets[0][1], w_final
        )
        return w_final, final_loss

    def get_convergence_history(self) -> List[float]:
        return self.history
