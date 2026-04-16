import numpy as np
import tensorflow as tf
from src.config import Config
from src.model_io import load_tinycnn_model

cfg = Config()

class MIFExporter:
    def __init__(self, bit_width=8, frac_bits=7):
        self.bit_width = bit_width
        self.frac_bits = frac_bits
        self.max_val = (2 ** (bit_width - 1)) - 1
        self.min_val = -(2 ** (bit_width - 1))

    def to_fixed_point(self, value):
        fixed_val = int(np.round(value * (2 ** self.frac_bits)))
        fixed_val = max(min(fixed_val, self.max_val), self.min_val)
        if fixed_val < 0:
            fixed_val = (1 << self.bit_width) + fixed_val
        return format(fixed_val, f'0{self.bit_width // 4}X')

    def generate_single_mif(self, model, filename="model_weights"):
        all_weights = []
        for layer in model.layers:
            weights = layer.get_weights()
            if not weights: continue
            all_weights.append(weights[0].flatten())
            all_weights.append(weights[1].flatten())
        
        flat_data = np.concatenate(all_weights)
        estimated_bytes = len(flat_data) * (self.bit_width // 8)
        if estimated_bytes > cfg.MAX_MIF_BYTES:
            raise RuntimeError(
                f"Pesos excedem limite da FPGA: {estimated_bytes} bytes > {cfg.MAX_MIF_BYTES} bytes."
            )

        mif_content = [
            f"DEPTH = {len(flat_data)};",
            f"WIDTH = {self.bit_width};",
            "ADDRESS_RADIX = HEX;",
            "DATA_RADIX = HEX;",
            "CONTENT BEGIN",
        ]
        for addr, val in enumerate(flat_data):
            mif_content.append(f"{addr:X} : {self.to_fixed_point(val)};")
        mif_content.append("END;")
        output_path = cfg.EXPORT_DIR / f"{filename}.mif"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(mif_content))
        print(f" -> Exportado: {filename}.mif ({len(flat_data)} words, {estimated_bytes} bytes)")

def export_model_to_mif():
    model_path = cfg.MODEL_PATH
    if not model_path.exists(): return
    model = load_tinycnn_model(model_path, compile_model=False)
    exporter = MIFExporter()
    print("\n[MIF] Iniciando exportação dos pesos quantizados num único arquivo...")
    exporter.generate_single_mif(model, "all_weights")
