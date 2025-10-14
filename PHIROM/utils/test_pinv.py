import jax
import jax.numpy as jnp
import lineax as lx
from jax.scipy.linalg import solve_triangular


# 1. QR-based approach
def apply_pinv_qr(A, b):
    Q, R = jnp.linalg.qr(A)
    y = Q.T @ b
    return solve_triangular(R, y)


# 2. Doing Pseudoinverse
def apply_pinv(A, b):
    return jnp.linalg.pinv(A) @ b


# 3. Least Squares solver
def apply_lstsq(A, b):
    return jnp.linalg.lstsq(A, b, rcond=1e-10)[0]


def apply_normalcg_functional(A, b):
    def F(x):
        return A @ x

    x = jnp.ones(A.shape[1])
    in_struc = jax.eval_shape(lambda: x)
    op = lx.FunctionLinearOperator(F, in_struc)
    solver = lx.NormalCG(1e-6, 1e-6)
    return lx.linear_solve(op, b, solver).value


def test_pseudoinverse_methods(m, n):
    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (m, n))  # Random Jacobian-like matrix
    b = jax.random.normal(key, (m,))  # Random vector

    # Compute A⁺ b using different methods
    x_qr = apply_pinv_qr(A, b)
    x_pinv = apply_pinv(A, b)
    x_lstsq = apply_lstsq(A, b)
    x_normalcg = apply_normalcg_functional(A, b)

    # Print differences
    print(f"||x_lstsq - x_pinv||_inf = {jnp.max(jnp.abs(x_lstsq - x_pinv))}")
    print(f"||x_qr - x_lstsq||_inf = {jnp.max(jnp.abs(x_qr - x_lstsq))}")
    print(f"||x_normalcg - x_pinv||_inf = {jnp.max(jnp.abs(x_normalcg - x_pinv))}")
    print(f"||x_normalcg - x_lstsq||_inf = {jnp.max(jnp.abs(x_normalcg - x_lstsq))}")


# --- Run Test ---
test_pseudoinverse_methods(m=256, n=2)  # Example: tall matrix
