import numpy as np
import tensorflow as tf
from src.config import Config

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

    def generate_mif(self, data, filename):
        flat_data = data.flatten()
        depth = len(flat_data)

        mif_content = [
            f"DEPTH = {depth};",
            f"WIDTH = {self.bit_width};",
            "ADDRESS_RADIX = HEX;",
            "DATA_RADIX = HEX;",
            "CONTENT",
            "BEGIN",
            ""
        ]

        for addr, val in enumerate(flat_data):
            hex_val = self.to_fixed_point(val)
            mif_content.append(f"{addr:X} : {hex_val};")

        mif_content.append("END;")

        output_path = cfg.PROJECT_ROOT / "export" / f"{filename}.mif"
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "w") as f:
            f.write("\n".join(mif_content))
        print(f" -> Exportado: {filename}.mif ({depth} words)")


def export_model_to_mif():
    model_path = cfg.PROJECT_ROOT / "models" / "tiny_cnn_binaria_final.h5"
    model = tf.keras.models.load_model(str(model_path))
    exporter = MIFExporter(bit_width=8, frac_bits=7)

    print("\n[MIF] Iniciando exportação dos pesos quantizados...")

    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue

        layer_name = layer.name
        # weights[0] são os pesos (filtros/kernel), weights[1] são os biases
        exporter.generate_mif(weights[0], f"{layer_name}_weights")
        exporter.generate_mif(weights[1], f"{layer_name}_biases")


if __name__ == "__main__":
    export_model_to_mif()