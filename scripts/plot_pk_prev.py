import csv
from pathlib import Path

def load_data(csv_path):
    data = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = int(row["layer"])
            val = float(row["P_K_prev"])
            data.setdefault(layer, []).append(val)
    return data

def make_svg(data, out_path, title):
    layers = sorted(data.keys())
    width = 1200
    padding = 60
    panel_height = 220
    plot_height = panel_height - 80
    plot_width = width - 2 * padding
    height = panel_height * len(layers) + padding

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    svg.append('<style>text { font-family: Arial, sans-serif; font-size: 18px; fill: #222; }</style>')
    svg.append(f'<text x="{width/2}" y="35" text-anchor="middle">{title}</text>')

    for idx, layer in enumerate(layers):
        values = data[layer]
        top = padding + idx * panel_height
        svg.append(f'<text x="{padding/2}" y="{top + 20}" text-anchor="middle" font-size="16">L{layer}</text>')
        svg.append(f'<rect x="{padding}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#888" stroke-width="1"/>')
        max_val = max(values) if values else 1.0
        min_val = min(values) if values else 0.0
        if max_val == min_val:
            max_val = min_val + 1.0
        n = len(values)
        pts = []
        for i, v in enumerate(values):
            x = padding + (i / (n - 1 if n > 1 else 1)) * plot_width
            norm = (v - min_val) / (max_val - min_val)
            y = top + plot_height - norm * plot_height
            pts.append(f"{x:.2f},{y:.2f}")
        color = f"hsl({(idx * 67) % 360}, 70%, 45%)"
        points = ' '.join(pts)
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<text x="{width - padding}" y="{top - 5}" text-anchor="end" font-size="14">max={max(values):.4g}</text>')
        svg.append(f'<text x="{width - padding}" y="{top + plot_height + 20}" text-anchor="end" font-size="14">min={min(values):.4g}</text>')

    svg.append('</svg>')
    Path(out_path).write_text('\n'.join(svg))

def main():
    base = Path('logs')
    specs = [
        ('pk_prev_counts_alpha.csv', 'pk_prev_counts_alpha.svg', 'P_K_prev vs edits (Alpha)'),
        ('pk_prev_counts.csv', 'pk_prev_counts.svg', 'P_K_prev vs edits (Baseline)'),
    ]
    for csv_name, out_name, title in specs:
        data = load_data(base / csv_name)
        make_svg(data, base / out_name, title)
        print(f'Wrote {base/out_name}')

if __name__ == '__main__':
    main()
