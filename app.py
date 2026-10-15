
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ---------------- REACTOR MODEL ----------------
class MembraneReactor:
    def __init__(self, kc=0.2, cto=0.2, k=0.7, Kc=0.05):
        self.kc = kc
        self.cto = cto
        self.k = k
        self.Kc = Kc
        self.sol = None

    def reaction_model(self, v, f):
        fa, fb, fc = f
        ft = fa + fb + fc

        if ft <= 1e-12:
            return [0.0, 0.0, 0.0]

        ya, yb, yc = fa / ft, fb / ft, fc / ft

        ra = -self.k * self.cto * (
            ya - (self.cto / self.Kc) * yb * yc
        )

        dfadv = ra
        dfbdv = -ra - self.kc * self.cto * yb
        dfcdv = -ra

        return [dfadv, dfbdv, dfcdv]

    def solve(self, v_end, initial_flows, n_points=200):
        self.sol = solve_ivp(
            self.reaction_model,
            (0, v_end),
            initial_flows,
            t_eval=np.linspace(0, v_end, n_points),
            method="RK45",
            rtol=1e-6,
            atol=1e-8
        )

        if not self.sol.success:
            raise RuntimeError(self.sol.message)

        return self.sol


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Membrane Reactor Simulator",
    page_icon="⚗️",
    layout="wide"
)

st.title("⚗️ Membrane Reactor Simulator")
st.caption("Chemical Engineering | OOP | SciPy ODE Solver")

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("Reactor Parameters")

with st.sidebar.form("reactor_inputs"):
    st.subheader("Kinetic Parameters")
    kc = st.number_input("kc", value=0.2, min_value=0.0)
    cto = st.number_input("cto", value=0.2, min_value=0.000001)
    k = st.number_input("k", value=0.7, min_value=0.0)
    Kc = st.number_input("Kc", value=0.05, min_value=0.000001)

    st.subheader("Initial Molar Flow Rates")
    fa0 = st.number_input("Initial fa", value=10.0, min_value=0.0)
    fb0 = st.number_input("Initial fb", value=0.0, min_value=0.0)
    fc0 = st.number_input("Initial fc", value=0.0, min_value=0.0)

    st.subheader("Simulation Settings")
    v_end = st.number_input("Final reactor volume", value=500.0, min_value=0.001)
    n_points = st.number_input(
        "Number of points", value=200, min_value=2, max_value=10000, step=50
    )

    run = st.form_submit_button("▶ Run Simulation", type="primary")

# ---------------- RUN SIMULATION ----------------
if run:
    if fa0 + fb0 + fc0 <= 0:
        st.error("At least one initial molar flow rate must be positive.")
    else:
        try:
            reactor = MembraneReactor(kc, cto, k, Kc)
            sol = reactor.solve(
                v_end, [fa0, fb0, fc0], int(n_points)
            )

            st.session_state["reactor_results"] = {
                "v": sol.t,
                "fa": sol.y[0],
                "fb": sol.y[1],
                "fc": sol.y[2],
                "parameters": {
                    "kc": kc, "cto": cto, "k": k, "Kc": Kc
                }
            }
        except Exception as e:
            st.error(f"Simulation failed: {e}")

# ---------------- DISPLAY RESULTS ----------------
if "reactor_results" in st.session_state:
    result = st.session_state["reactor_results"]

    v = result["v"]
    fa, fb, fc = result["fa"], result["fb"], result["fc"]

    st.subheader("Simulation Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Outlet fa", f"{fa[-1]:.5f}")
    col2.metric("Outlet fb", f"{fb[-1]:.5f}")
    col3.metric("Outlet fc", f"{fc[-1]:.5f}")

    # Flow-rate plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(v, fa, label="fa", linewidth=2)
    ax.plot(v, fb, label="fb", linewidth=2)
    ax.plot(v, fc, label="fc", linewidth=2)
    ax.set_xlabel("Reactor Volume, v")
    ax.set_ylabel("Molar Flow Rate")
    ax.set_title("Molar Flow Rates vs Reactor Volume")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # Results table
    st.subheader("Numerical Results")
    df = pd.DataFrame({
        "Reactor Volume": v,
        "fa": fa,
        "fb": fb,
        "fc": fc
    })

    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Results (CSV)",
        data=csv,
        file_name="membrane_reactor_results.csv",
        mime="text/csv"
    )

    st.success("Simulation completed successfully.")

else:
    st.info("Set your reactor parameters in the sidebar and click Run Simulation.")

st.divider()
st.caption("Membrane Reactor Simulator | Built with Streamlit, SciPy and Matplotlib")
