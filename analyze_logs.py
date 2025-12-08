import re
from pathlib import Path
import matplotlib.pyplot as plt

LOG_DIR = Path("logs")

# regex pra pegar ratio do nome: baseline_r0.3.log → 0.3
ratio_re = re.compile(r"r([0-9.]+)\.log$")
# regex pra pegar a última linha de Average Test Loss
loss_re = re.compile(r"Average Test Loss:\s*([0-9.]+)")

results = {}  # (ratio -> {"baseline": loss, "time": loss})

for log_path in sorted(LOG_DIR.glob("*.log")):
    text = log_path.read_text()

    # pega o ratio pelo nome
    m_ratio = ratio_re.search(log_path.name)
    if not m_ratio:
        continue
    ratio = float(m_ratio.group(1))

    # pega TODAS as ocorrências de Average Test Loss e usa a mínima
    losses = loss_re.findall(text)
    if not losses:
        print(f"[WARN] Nenhuma 'Average Test Loss' em {log_path.name}")
        continue

    last_loss = min(float(loss) for loss in losses)

    # baseline ou time?
    if "baseline" in log_path.name:
        key = "baseline"
    elif "time" in log_path.name:
        key = "time"
    else:
        key = "unknown"

    if ratio not in results:
        results[ratio] = {}
    results[ratio][key] = last_loss

# imprime tabelinha
print("ratio\tbaseline\t time")
for ratio in sorted(results.keys()):
    base = results[ratio].get("baseline", float("nan"))
    time = results[ratio].get("time", float("nan"))
    print(f"{ratio:.2f}\t{base:.6f}\t{time:.6f}")

# gera gráfico
ratios_sorted = sorted(results.keys())
baseline_losses = [results[r].get("baseline", float("nan"))
                   for r in ratios_sorted]
time_losses = [results[r].get("time", float("nan")) for r in ratios_sorted]

plt.figure(figsize=(6, 4))
plt.plot(ratios_sorted, baseline_losses, marker="o",
         label="Sem codificação temporal")
plt.plot(ratios_sorted, time_losses, marker="o",
         label="Com codificação temporal")
plt.xlabel("Proporção de dados removidos (DATA_REMOVAL_RATIO)")
plt.ylabel("Average Test Loss final (MSE)")
plt.title("Impacto da codificação temporal em dados faltantes")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("comparacao_codificacao_temporal.png")
plt.show()
