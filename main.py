import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk


class HSLConverter:
    """Ручные векторизованные преобразования RGB <-> HSL."""

    @staticmethod
    def rgb_to_hsl(rgb_uint8):
        """
        RGB [0..255] -> HSL.
        H: [0..360), S: [0..1], L: [0..1].
        """
        rgb = rgb_uint8.astype(np.float32) / 255.0
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]

        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        h = np.zeros_like(cmax, dtype=np.float32)
        non_gray = delta != 0

        r_is_max = (cmax == r) & non_gray
        g_is_max = (cmax == g) & non_gray
        b_is_max = (cmax == b) & non_gray

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
    def hsl_to_rgb(hsl):
        """
        HSL -> RGB.
        H: [0..360), S: [0..1], L: [0..1].
        Результат: RGB uint8 [0..255].
        """
        h = (hsl[..., 0] % 360.0) / 360.0
        s = np.clip(hsl[..., 1], 0.0, 1.0)
        l = np.clip(hsl[..., 2], 0.0, 1.0)

        q = np.where(l < 0.5, l * (1.0 + s), l + s - l * s)
        p = 2.0 * l - q

        t_r = h + 1.0 / 3.0
        t_g = h
        t_b = h - 1.0 / 3.0

        rgb_float = np.stack(
            [
                HSLConverter._hue_to_channel(p, q, t_r),
                HSLConverter._hue_to_channel(p, q, t_g),
                HSLConverter._hue_to_channel(p, q, t_b),
            ],
            axis=-1,
        )

        return np.clip(np.round(rgb_float * 255.0), 0, 255).astype(np.uint8)

    @staticmethod
    def _hue_to_channel(p, q, t):
        """Вспомогательная функция для HSL -> RGB по формуле через P, Q и T."""
        t = t.copy()
        t[t < 0.0] += 1.0
        t[t > 1.0] -= 1.0

        channel = np.empty_like(t, dtype=np.float32)

        mask1 = t < 1.0 / 6.0
        mask2 = (t >= 1.0 / 6.0) & (t < 1.0 / 2.0)
        mask3 = (t >= 1.0 / 2.0) & (t < 2.0 / 3.0)
        mask4 = t >= 2.0 / 3.0

        channel[mask1] = p[mask1] + ((q[mask1] - p[mask1]) * 6.0 * t[mask1])
        channel[mask2] = q[mask2]
        channel[mask3] = p[mask3] + ((q[mask3] - p[mask3]) * (2.0 / 3.0 - t[mask3]) * 6.0)
        channel[mask4] = p[mask4]

        return channel


class HSLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RGB ↔ HSL")

        self.original_image = None
        self.original_rgb = None
        self.original_hsl = None
        self.current_rgb = None
        self.preview_image = None
        self.update_job = None
        self.detected_cast_hue = None

        self.mode_var = tk.StringVar(value="hsl")
        self.hue_var = tk.DoubleVar(value=0)
        self.sat_var = tk.DoubleVar(value=100)
        self.light_var = tk.DoubleVar(value=100)
        self.white_balance_strength_var = tk.DoubleVar(value=50)

        self._build_interface()

    def _build_interface(self):
        self.image_label = tk.Label(self.root, text="Загрузите изображение PNG, JPG/JPEG или BMP")
        self.image_label.pack(padx=10, pady=10)

        mode_frame = tk.LabelFrame(self.root, text="Режим работы")
        mode_frame.pack(fill="x", padx=10, pady=5)

        tk.Radiobutton(
            mode_frame,
            text="Обычная коррекция HSL",
            variable=self.mode_var,
            value="hsl",
            command=self._on_mode_change,
        ).pack(side="left", padx=8, pady=5)

        tk.Radiobutton(
            mode_frame,
            text="Баланс белого через HSL",
            variable=self.mode_var,
            value="white_balance",
            command=self._on_mode_change,
        ).pack(side="left", padx=8, pady=5)

        self.hsl_frame = tk.LabelFrame(self.root, text="Параметры HSL")
        self.hsl_frame.pack(fill="x", padx=10, pady=5)

        self._add_slider(self.hsl_frame, "Hue, сдвиг тона (°)", self.hue_var, -180, 180)
        self._add_slider(self.hsl_frame, "Saturation, масштаб (%)", self.sat_var, 0, 200)
        self._add_slider(self.hsl_frame, "Lightness, масштаб (%)", self.light_var, 0, 200)

        self.white_balance_frame = tk.LabelFrame(self.root, text="Баланс белого через HSL")

        self.cast_label = tk.Label(
            self.white_balance_frame,
            text="Преобладающий цветовой тон будет определён после загрузки изображения.",
            anchor="w",
        )
        self.cast_label.pack(fill="x", padx=8, pady=5)

        self._add_slider(
            self.white_balance_frame,
            "Сила коррекции (%)",
            self.white_balance_strength_var,
            0,
            100,
        )

        tk.Button(
            self.white_balance_frame,
            text="Пересчитать цветовой оттенок",
            command=self.recalculate_cast_hue,
        ).pack(side="left", padx=8, pady=8)

        buttons = tk.Frame(self.root)
        buttons.pack(fill="x", padx=10, pady=10)

        tk.Button(buttons, text="Открыть изображение", command=self.open_image).pack(side="left", padx=4)
        tk.Button(buttons, text="Сбросить", command=self.reset_controls).pack(side="left", padx=4)
        tk.Button(buttons, text="Сохранить результат", command=self.save_image).pack(side="left", padx=4)

    def _add_slider(self, parent, label, variable, min_value, max_value):
        frame = tk.Frame(parent)
        frame.pack(fill="x", pady=3, padx=8)

        tk.Label(frame, text=label, width=28, anchor="w").pack(side="left")
        slider = tk.Scale(
            frame,
            from_=min_value,
            to=max_value,
            orient="horizontal",
            variable=variable,
            command=lambda _value: self.schedule_update(),
        )
        slider.pack(side="left", fill="x", expand=True)

    def _on_mode_change(self):
        if self.mode_var.get() == "hsl":
            self.white_balance_frame.pack_forget()
            self.hsl_frame.pack(fill="x", padx=10, pady=5, after=self.root.children[list(self.root.children.keys())[1]])
        else:
            self.hsl_frame.pack_forget()
            self.white_balance_frame.pack(fill="x", padx=10, pady=5, after=self.root.children[list(self.root.children.keys())[1]])
            self.recalculate_cast_hue(update_after=False)

        self.schedule_update()

    def open_image(self):
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
        self.detected_cast_hue = self._detect_cast_hue(self.original_hsl)
        self._update_cast_label()
        self.reset_controls()

    def reset_controls(self):
        self.hue_var.set(0)
        self.sat_var.set(100)
        self.light_var.set(100)
        self.white_balance_strength_var.set(50)
        self.schedule_update()

    def recalculate_cast_hue(self, update_after=True):
        if self.original_hsl is None:
            return
        self.detected_cast_hue = self._detect_cast_hue(self.original_hsl)
        self._update_cast_label()
        if update_after:
            self.schedule_update()

    def _update_cast_label(self):
        if self.detected_cast_hue is None:
            self.cast_label.config(text="Цветовой оттенок не определён: изображение почти ахроматическое.")
        else:
            opposite = (self.detected_cast_hue + 180.0) % 360.0
            self.cast_label.config(
                text=(
                    f"Определён преобладающий оттенок: {self.detected_cast_hue:.1f}°. "
                    f"Коррекция выполняется в сторону {opposite:.1f}° и снижает насыщенность оттенка."
                )
            )

    def schedule_update(self):
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(80, self.update_image)

    def update_image(self):
        self.update_job = None
        if self.original_hsl is None:
            return

        if self.mode_var.get() == "hsl":
            hsl = self._apply_manual_hsl_correction()
        else:
            hsl = self._apply_white_balance_correction()

        self.current_rgb = HSLConverter.hsl_to_rgb(hsl)
        image = Image.fromarray(self.current_rgb, "RGB")

        image.thumbnail((900, 600))
        self.preview_image = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.preview_image, text="")

    def _apply_manual_hsl_correction(self):
        hsl = self.original_hsl.copy()
        hsl[..., 0] = (hsl[..., 0] + self.hue_var.get()) % 360.0
        hsl[..., 1] = np.clip(hsl[..., 1] * (self.sat_var.get() / 100.0), 0.0, 1.0)
        hsl[..., 2] = np.clip(hsl[..., 2] * (self.light_var.get() / 100.0), 0.0, 1.0)
        return hsl

    def _apply_white_balance_correction(self):
        hsl = self.original_hsl.copy()

        if self.detected_cast_hue is None:
            return hsl

        strength = self.white_balance_strength_var.get() / 100.0
        cast_hue = self.detected_cast_hue
        opposite_hue = (cast_hue + 180.0) % 360.0

        hue = hsl[..., 0]
        saturation = hsl[..., 1]

        distance_to_cast = np.abs(self._angle_delta(hue, cast_hue))
        closeness = np.clip(1.0 - distance_to_cast / 180.0, 0.0, 1.0)

        shift_to_opposite = self._angle_delta(opposite_hue, hue)
        hsl[..., 0] = (hue + shift_to_opposite * closeness * strength * 0.25) % 360.0
        hsl[..., 1] = np.clip(saturation * (1.0 - 0.45 * closeness * strength), 0.0, 1.0)

        return hsl

    @staticmethod
    def _detect_cast_hue(hsl):
        """
        Определение преобладающего цветового оттенка.
        Используются достаточно яркие и насыщенные пиксели.
        Если таких пикселей мало, используется более общий набор цветных пикселей.
        """
        hue = hsl[..., 0]
        saturation = hsl[..., 1]
        lightness = hsl[..., 2]

        mask = (saturation > 0.08) & (lightness > 0.45) & (lightness < 0.95)
        if np.count_nonzero(mask) < 100:
            mask = saturation > 0.08
        if np.count_nonzero(mask) == 0:
            return None

        selected_hue = hue[mask]
        weights = saturation[mask] * np.maximum(lightness[mask], 0.1)

        radians = np.deg2rad(selected_hue)
        mean_sin = np.sum(np.sin(radians) * weights)
        mean_cos = np.sum(np.cos(radians) * weights)

        if abs(mean_sin) < 1e-6 and abs(mean_cos) < 1e-6:
            return None

        result = np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360.0
        return float(result)

    @staticmethod
    def _angle_delta(target, source):
        """Кратчайшая угловая разница target - source в диапазоне [-180; 180]."""
        return (target - source + 180.0) % 360.0 - 180.0

    def save_image(self):
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


def main():
    root = tk.Tk()
    HSLApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
