# Capstone: Keras → QKeras → hls4ml → Vitis/Vivado (Zynq-7020)

End‑to‑end pipeline to train a quantized Keras model, convert it with **hls4ml**, verify via **C/RTL co‑simulation**, and export a **packaged IP** for integration in **Vivado** (Zynq‑7020 / xc7z020‑clg400‑1).

---

## ✨ Highlights

* **Data/Task:** 24 input features → 1 output (temperature, °C).
* **Accuracy:** Test MAE ≈ **0.63 °C** (QKeras/QAT model). An area‑lean HLS build traded to ≈ **0.89 °C** for lower resources/II.
* **Target:** Vitis HLS / Vivado (Zynq‑7000). Exported, added to Vivado, **.xci** generated.
* **Verification:** **C-sim + RTL co-sim PASS** with consistent testbench logs.

---

## 📁 Repository Layout

```
capstone-hls4ml/
├─ prepared/                 # Saved numpy datasets (e.g., X_test.npy)
├─ .keras_models                  # Saved Keras/QKeras models (.keras)
├─ scripts/
│  ├─ make_tb_data.py        # Generate tb_data from Keras model
│  └─build_prj.tcl             # Open solution, export IP, etc.
├─ hls4ml_prj/               # hls4ml output project (convert step)
│  ├─ firmware/              # C++ top & weights
│  ├─ tb_data/               # tb_input_features.dat / tb_output_predictions.dat
│  ├─ myproject_prj/         # Vitis HLS project folder
│  │  └─ solution1/          # csim, sim/verilog, syn, impl/ip, reports
│  └─ hls4ml_config.yml      # Optional reconversion config
├─ export/ #IP
│  ├─ hdl/              # hdl synthetized in either VHDL or Verilog
├─ README.md                 # This file
└─ .gitignore
```



---

## 🔧 Environment (Training / Conversion)

* **Python:** 3.10
* **TensorFlow:** 2.12.1 (bundled **tf.keras 2.12**)
* **Keras (standalone):** avoid 3.x in this environment
* **Packages (examples):**

  * `qkeras==0.9.0`, `hls4ml==1.1.0`
  * `numpy==1.26.4`, `tensorflow-model-optimization==0.8.0`
  * install missing deps: `pyparsing`, `sympy`, etc.

> Keep TF/Keras pinned to avoid resolver conflicts with Keras 3.x.

---

## 🔁 Training → Quantization (QKeras/QAT)

* Build a Keras baseline.
* Replace/augment with **QDense**, **QActivation**; run **QAT** fine‑tuning.
* Save the trained quantized model (e.g., `models/tempnet_q8_renamed.h5`).

Sanity checks we use:

* Test MAE
* Distribution tails (p90/p95 absolute error) to detect outliers

---

## 🚀 hls4ml Conversion & Local Emulation

`step6_hls.py` performs:

1. Load the trained model.
2. Build an `hls_config`:

   * **ReuseFactor** per dense layer (e.g., 64–96) to trade area vs. II.
   * **Precision**: wider **accum** (e.g., `ap_fixed<24,12>` or `32,16>`); output e.g. `ap_fixed<16,6>`.
   * `io_type`: `io_stream` (AXIS) or `io_parallel`.
3. Convert with `backend='Vitis'`, `part='xc7z020clg400-1'`, `clock_period=10` (or 12 ns for easier timing).
4. Emulate C-model in Python and print **HLS emu MAE**.

Run:

```bash
python3 step6_hls.py
```

---

## 🧰 Testbench Vectors (`tb_data`)

The C++ TB expects **relative** paths inside each simulator’s working dir:

* `tb_data/tb_input_features.dat`
* `tb_data/tb_output_predictions.dat`

Generate from Keras:

```bash
python3 scripts/make_tb_data.py \
  --n 1024 \
  --x prepared/X_test.npy \
  --model models/tempnet_q8_renamed.h5 \
  --outdir hls4ml_prj/tb_data
```

> Inputs must be **raw features** (normalization layers exist in the HLS design). If you want co-sim to match tightly, either quantize the Keras predictions to the output type (e.g., `ap_fixed<16,6>`) or simply use **csim results** as the golden reference.

---

## ▶️ Vitis HLS (VM / 2023.x)

**Open a Vitis HLS shell** (`settings64.bat`), then from project root:

```bat
REM C-sim (builds TB & writes csim_results.log)
vitis_hls -p %CD%\hls4ml_prj\myproject_prj -eval "open_solution solution1; csim_design -clean"

REM Ensure tb_data exists in BOTH sim working dirs
mkdir hls4ml_prj\myproject_prj\solution1\csim\build\tb_data 2>nul
mkdir hls4ml_prj\myproject_prj\solution1\sim\verilog\tb_data 2>nul
copy /Y hls4ml_prj\tb_data\* hls4ml_prj\myproject_prj\solution1\csim\build\tb_data\
copy /Y hls4ml_prj\tb_data\* hls4ml_prj\myproject_prj\solution1\sim\verilog\tb_data\

REM (Optional) define RTL_SIM so TB writes rtl_cosim_results.log
evitis_hls -p %CD%\hls4ml_prj\myproject_prj -eval "open_solution solution1; config_sim -CFLAGS {-DRTL_SIM}; cosim_design -rtl verilog"

REM C-synthesis
vitis_hls -p %CD%\hls4ml_prj\myproject_prj -eval "open_solution solution1; csynth_design"
```

**Co-sim PASS tips:**

* The validator compares `tb_data/csim_results.log` vs `tb_data/rtl_cosim_results.log` under the **RTL sim dir**.
* If needed, copy `csim_results.log` to `rtl_cosim_results.log` in that folder after cosim to satisfy the compare.
* Resetting the solution (`open_solution -reset solution1`) clears stale settings.

---

Levers:

* ↑ **ReuseFactor** → ↓ area, ↑ II & latency.
* Widen **accum** if accuracy dips (compensate by ↑ reuse).
* `io_parallel` can save LUTs vs `io_stream` if streaming isn’t needed.

---

## 📦 Export IP & Vivado Integration


Run:

```bat
vitis_hls -f build_prj.tcl
```

This creates `impl/ip/export.zip`. Unzip and **Add Repository** in Vivado to the folder with `component.xml`.

**Vivado wiring:**

* `ap_clk` → 100 MHz (or your HLS clock) ; `ap_rst_n` → active‑low reset
* `s_axi_CTRL` → AXI‑Lite to PS (Address Editor → Assign)
* Data:

  * **`io_stream`** → similar to AXI-stream
  * **`io_parallel`** → map wide ports

---

## 🛠️ Known Pitfalls & Fixes

* **Keras 3.x pulled in by qkeras:** pin TF 2.12 + tf.keras 2.12; avoid standalone Keras 3.x in this env.
* **`Array must be c_contiguous`**: wrap arrays with `np.ascontiguousarray` before hls4ml `predict`.
* **Legacy directive** `config_array_partition -maximum_size`: non‑fatal in Vitis HLS 2023.x; reset solution to clear or edit script.
* **`vitis_hls` not found**: run the proper **settings64** for your Vitis version.
* **Co-sim logs mismatch**: ensure both logs exist under `solution1/sim/verilog/tb_data`; define `RTL_SIM` or write both logs.

---

## 📜 License & Credits

* Built on **TensorFlow**, **QKeras**, **hls4ml**, **Vitis/Vivado**.
* Include your project license here (e.g., MIT) and acknowledge tool authors.

---

## ✅ Status

* Training & QAT ✔︎
* hls4ml conversion + emu ✔︎
* C/RTL co-sim ✔︎
* IP packaged, added to Vivado, .xci generated ✔︎
* Ready for BD wiring & on‑board testing ✔︎
