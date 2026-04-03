import pandas as pd
import json
import sys
from pathlib import Path
from tabulate import tabulate
from src.config import Config

def analisar_hiperparametros(tuner_dir=None, min_trials=1):
    cfg = Config()
    
    # Se nenhum diretório foi passado ou ele não existe, tenta encontrar o mais recente em tuner_logs
    if tuner_dir is None or not Path(tuner_dir).exists():
        logs_dir = cfg.TUNER_LOGS_DIR
        subdirs = [d for d in logs_dir.iterdir() if d.is_dir() and list(d.glob("trial_*"))]
        
        if not subdirs:
            print(f"Erro: Nenhum diretório de tuner com trials encontrado em '{logs_dir}'. J rode o treinamento primeiro.")
            sys.exit(1)
            
        # Pega a pasta modificada por último para ser a padrão
        path = max(subdirs, key=lambda d: d.stat().st_mtime)
        print(f"Usando o diretório de tuner mais recente encontrado: {path.name}")
    else:
        path = Path(tuner_dir)

    trials_data = []

    # Itera sobre todos os diretórios "trial_..." gerados pelo keras-tuner 
    for trial_dir in path.glob("trial_*"):
        trial_json = trial_dir / "trial.json"
        if trial_json.exists():
            with open(trial_json, "r") as f:
                data = json.load(f)

            hiperparametros = data.get("hyperparameters", {}).get("values", {})
            metrics = data.get("metrics", {}).get("metrics", {})
            
            # Pega as métricas de validação (se existirem, pois trials que falharam não terão)
            # A estrutura do Keras Tuner salva as métricas de maneiras ligeiramente diferentes
            # dependendo da versão. Pode ser ['metrics']['val_auc']['observations'][0]['value'][0] 
            # na estrutura completa.
            try:
                val_auc = metrics.get("val_auc", {}).get("observations", [{}])[0].get("value", [None])[0]
                val_precision = metrics.get("val_precision", {}).get("observations", [{}])[0].get("value", [None])[0]
                val_loss = metrics.get("val_loss", {}).get("observations", [{}])[0].get("value", [None])[0]
            except (IndexError, KeyError):
                val_auc = None

            status = data.get("status", "UNKNOWN")

            if status == "COMPLETED" and val_auc is not None:
                record = {
                    "Trial ID": data.get("trial_id", ""),
                    "L2 Reg": hiperparametros.get("l2_reg"),
                    "Dropout": hiperparametros.get("dropout"),
                    "Otimizador": hiperparametros.get("optimizer"),
                    "Learn Rate": hiperparametros.get("learning_rate"),
                    "Peso C0": hiperparametros.get("peso_classe_0"),
                    "Val AUC": val_auc,
                    "Val Precision": val_precision,
                    "Val Loss": val_loss
                }
                trials_data.append(record)

    if len(trials_data) < min_trials:
        print("Nenhum dado de Trial concluído encontrado. Rode o treinamento primeiro.")
        sys.exit(0)

    # Cria DataFrame e ordena pela melhor AUC
    df = pd.DataFrame(trials_data)
    df = df.sort_values(by="Val AUC", ascending=False).reset_index(drop=True)

    print("\n" + "="*80)
    print("      🏆 RANKING DAS ARQUITETURAS EXPLORADAS (KERAS TUNER)      ")
    print("="*80)
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))

    # Salva os resultados em um CSV na pasta reports/ para análise futura
    cfg = Config()
    report_csv = cfg.PROJECT_ROOT / "reports" / "historico_hiperparametros.csv"
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_csv, index=False)
    
    print(f"\n💾 Tabela completa salva com sucesso em: {report_csv}")

if __name__ == "__main__":
    cfg = Config()
    # Tenta usar o nome oficial, se não achar, a função buscará o mais recente que houver na pasta
    tuner_path = cfg.TUNER_LOGS_DIR / "tiny_cnn_search"
    analisar_hiperparametros(tuner_path)
