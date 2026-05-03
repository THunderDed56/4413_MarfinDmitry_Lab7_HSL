import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk


class HSLConverter:
    @staticmethod
    def rgb_to_hsl(rgb_uint8: np.ndarray) -> np.ndarray:
        """
        RGB [0..255] -> HSL:
        H в градусах [0..360), S и L в диапазоне [0..1].
        """
        rgb = rgb_uint8.astype(np.float32) / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        h = np.zeros_like(cmax, dtype=np.float32)

        mask = delta != 0

        r_is_max = (cmax == r) & mask
        g_is_max = (cmax == g) & mask
        b_is_max = (cmax == b) & mask

        h[r_is_max] = (60.0 * ((g[r_is_max] - b[r_is_max]) / delta[r_is_max])) % 360.0
        h[g_is_max] = 60.0 * ((b[g_is_max] - r[g_is_max]) / delta[g_is_max] + 2.0)
        h[b_is_max] = 60.0 * ((r[b_is_max] - g[b_is_max]) / delta[b_is_max] + 4.0)

        l = (cmax + cmin) / 2.0

        s = np.zeros_like(cmax, dtype=np.float32)
        s_mask = delta != 0
        denominator = 1.0 - np.abs(2.0 * l[s_mask] - 1.0)
        s[s_mask] = delta[s_mask] / denominator

        return np.stack([h, s, l], axis=-1)

    @staticmethod
    def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
        """
        HSL -> RGB:
        H в градусах [0..360), S и L в диапазоне [0..1],
        результат RGB uint8 [0..255].
        """
        h = hsl[..., 0] % 360.0
        s = np.clip(hsl[..., 1], 0.0, 1.0)
        l = np.clip(hsl[..., 2], 0.0, 1.0)

        c = (1.0 - np.abs(2.0 * l - 1.0)) * s
        h_prime = h / 60.0
        x = c * (1.0 - np.abs((h_prime % 2.0) - 1.0))
        m = l - c / 2.0

        r1 = np.zeros_like(h)
        g1 = np.zeros_like(h)
        b1 = np.zeros_like(h)

        masks = [
            (0 <= h_prime) & (h_prime < 1),
            (1 <= h_prime) & (h_prime < 2),
            (2 <= h_prime) & (h_prime < 3),
            (3 <= h_prime) & (h_prime < 4),
            (4 <= h_prime) & (h_prime < 5),
            (5 <= h_prime) & (h_prime < 6),
        ]

        r1[masks[0]], g1[masks[0]], b1[masks[0]] = c[masks[0]], x[masks[0]], 0
        r1[masks[1]], g1[masks[1]], b1[masks[1]] = x[masks[1]], c[masks[1]], 0
        r1[masks[2]], g1[masks[2]], b1[masks[2]] = 0, c[masks[2]], x[masks[2]]
        r1[masks[3]], g1[masks[3]], b1[masks[3]] = 0, x[masks[3]], c[masks[3]]
        r1[masks[4]], g1[masks[4]], b1[masks[4]] = x[masks[4]], 0, c[masks[4]]
        r1[masks[5]], g1[masks[5]], b1[masks[5]] = c[masks[5]], 0, x[masks[5]]

        rgb = np.stack([r1 + m, g1 + m, b1 + m], axis=-1)
        return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


class HSLApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("RGB ↔ HSL")

        self.original_image: Image.Image | None = None
        self.original_rgb: np.ndarray | None = None
        self.original_hsl: np.ndarray | None = None
        self.current_rgb: np.ndarray | None = None
        self.preview_image: ImageTk.PhotoImage | None = None

        self.image_label = tk.Label(root, text="Загрузите изображение PNG, JPEG или BMP")
        self.image_label.pack(padx=10, pady=10)

        controls = tk.Frame(root)
        controls.pack(fill="x", padx=10)

        self.hue_var = tk.DoubleVar(value=0)
        self.sat_var = tk.DoubleVar(value=100)
        self.light_var = tk.DoubleVar(value=100)

        self._add_slider(controls, "Hue, сдвиг тона (°)", self.hue_var, -180, 180)
        self._add_slider(controls, "Saturation, масштаб (%)", self.sat_var, 0, 200)
        self._add_slider(controls, "Lightness, масштаб (%)", self.light_var, 0, 200)

        buttons = tk.Frame(root)
        buttons.pack(fill="x", padx=10, pady=10)

        tk.Button(buttons, text="Открыть изображение", command=self.open_image).pack(side="left", padx=4)
        tk.Button(buttons, text="Сбросить", command=self.reset_sliders).pack(side="left", padx=4)
        tk.Button(buttons, text="Сохранить результат", command=self.save_image).pack(side="left", padx=4)

    def _add_slider(self, parent: tk.Frame, label: str, variable: tk.DoubleVar, min_value: int, max_value: int) -> None:
        frame = tk.Frame(parent)
        frame.pack(fill="x", pady=3)
        tk.Label(frame, text=label, width=28, anchor="w").pack(side="left")
        slider = tk.Scale(
            frame,
            from_=min_value,
            to=max_value,
            orient="horizontal",
            variable=variable,
            command=lambda _value: self.update_image(),
        )
        slider.pack(side="left", fill="x", expand=True)

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.png *.jpg *.jpeg *.bmp"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("BMP", "*.bmp"),
            ],
        )
        if not path:
            return

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{exc}")
            return

        self.original_image = image
        self.original_rgb = np.array(image)
        self.original_hsl = HSLConverter.rgb_to_hsl(self.original_rgb)
        self.reset_sliders()

    def reset_sliders(self) -> None:
        self.hue_var.set(0)
        self.sat_var.set(100)
        self.light_var.set(100)
        self.update_image()

    def update_image(self) -> None:
        if self.original_hsl is None:
            return

        hsl = self.original_hsl.copy()
        hsl[..., 0] = (hsl[..., 0] + self.hue_var.get()) % 360.0
        hsl[..., 1] = np.clip(hsl[..., 1] * (self.sat_var.get() / 100.0), 0.0, 1.0)
        hsl[..., 2] = np.clip(hsl[..., 2] * (self.light_var.get() / 100.0), 0.0, 1.0)

        self.current_rgb = HSLConverter.hsl_to_rgb(hsl)
        image = Image.fromarray(self.current_rgb, "RGB")

        image.thumbnail((900, 600))
        self.preview_image = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.preview_image, text="")

    def save_image(self) -> None:
        if self.current_rgb is None:
            messagebox.showwarning("Нет изображения", "Сначала загрузите изображение.")
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("BMP", "*.bmp"),
            ],
        )
        if not path:
            return

        try:
            Image.fromarray(self.current_rgb, "RGB").save(path)
            messagebox.showinfo("Готово", f"Файл сохранён:\n{Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{exc}")


def main() -> None:
    root = tk.Tk()
    app = HSLApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
