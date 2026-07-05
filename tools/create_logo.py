"""
SIMPLE_AI Logo Generator
Creates a modern, professional app icon and ICO file.
Design: Rounded gradient background with a stylized neural/AI mark.
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math, os

SIZE = 1024  # master size, downscaled for ICO

def lerp_color(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

def draw_gradient(img, c_top_left, c_bottom_right):
    """Diagonal gradient fill."""
    px = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            t = (x / w * 0.5 + y / h * 0.5)
            px[x, y] = lerp_color(c_top_left, c_bottom_right, t)

def rounded_rect_mask(size, radius):
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, size[0]-1, size[1]-1], radius=radius, fill=255)
    return mask

def draw_node(draw, cx, cy, r, color=(255, 255, 255)):
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)

def draw_thick_line(draw, x1, y1, x2, y2, width, color):
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

def draw_glow_circle(img, cx, cy, r, color, intensity=0.5):
    """Draw a soft glowing circle."""
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for i in range(int(r * 2), 0, -1):
        alpha = int(intensity * 255 * (1 - i / (r * 2)) ** 2)
        c = color + (min(alpha, 255),)
        gd.ellipse([cx - i, cy - i, cx + i, cy + i], fill=c)
    img.paste(Image.alpha_composite(Image.new("RGBA", img.size, (0, 0, 0, 0)), glow), (0, 0), glow)
    return img

def create_logo():
    S = SIZE
    # -- Background: diagonal gradient indigo → violet-purple --
    bg = Image.new("RGB", (S, S), (0, 0, 0))
    draw_gradient(bg, (15, 12, 60), (110, 50, 200))  # deep indigo → vivid purple

    # Convert to RGBA
    img = bg.convert("RGBA")

    # -- Rounded corners mask --
    corner_r = int(S * 0.22)
    mask = rounded_rect_mask((S, S), corner_r)
    
    # Apply rounded corners (transparent outside)
    result = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    result.paste(img, (0, 0), mask)
    img = result

    # -- Subtle inner glow (center light) --
    img = draw_glow_circle(img, int(S * 0.45), int(S * 0.40), int(S * 0.5),
                           (140, 100, 255), intensity=0.15)

    # -- Neural network nodes & connections --
    # Layout: stylized "S" curve made of connected nodes
    # Top cluster → middle → bottom cluster forming an S-shape

    draw = ImageDraw.Draw(img)

    # Node positions forming an elegant S-curve / neural path
    nodes = [
        # Top layer (3 nodes)
        (0.62, 0.20),  # top-right
        (0.42, 0.22),  # top-center
        (0.28, 0.28),  # top-left

        # Middle layer (3 nodes) — shifted right
        (0.38, 0.42),  # mid-left
        (0.52, 0.50),  # center (main hub)
        (0.68, 0.46),  # mid-right

        # Bottom layer (3 nodes)
        (0.72, 0.62),  # bottom-right
        (0.56, 0.70),  # bottom-center
        (0.36, 0.74),  # bottom-left
    ]

    # Convert to pixel coords
    pts = [(int(x * S), int(y * S)) for x, y in nodes]

    # Connections (index pairs) — forming the S-flow
    connections = [
        (0, 1), (1, 2),        # top row
        (0, 4), (0, 5),        # top to mid
        (1, 3), (1, 4),        # top to mid
        (2, 3),                 # top-left to mid-left
        (3, 4), (4, 5),        # mid row
        (3, 8),                 # mid-left to bottom-left
        (4, 7), (4, 6),        # center to bottom
        (5, 6),                 # mid-right to bottom-right
        (7, 8), (6, 7),        # bottom row
    ]

    # Draw connections (semi-transparent white lines)
    line_color = (255, 255, 255, 90)
    line_w = max(4, int(S * 0.005))
    
    # Create a separate layer for lines
    line_layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ld = ImageDraw.Draw(line_layer)
    for a, b in connections:
        ld.line([pts[a], pts[b]], fill=(255, 255, 255, 70), width=line_w)
    
    img = Image.alpha_composite(img, line_layer)
    draw = ImageDraw.Draw(img)

    # Draw node glows first
    for i, (px, py) in enumerate(pts):
        glow_r = int(S * 0.04)
        if i == 4:  # center hub — bigger glow
            glow_r = int(S * 0.06)
        img = draw_glow_circle(img, px, py, glow_r, (200, 180, 255), intensity=0.3)

    draw = ImageDraw.Draw(img)

    # Draw nodes (white circles with slight size variation)
    for i, (px, py) in enumerate(pts):
        if i == 4:  # center hub node — largest
            r = int(S * 0.032)
            draw_node(draw, px, py, r, (255, 255, 255))
            # Inner accent
            draw_node(draw, px, py, int(r * 0.5), (180, 140, 255))
        else:
            r = int(S * 0.018)
            draw_node(draw, px, py, r, (255, 255, 255))

    # -- "SIMPLE" text at bottom --
    # Try to use a clean font, fall back to default
    text_layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    td = ImageDraw.Draw(text_layer)

    font_size = int(S * 0.09)
    try:
        font = ImageFont.truetype("segoeui.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

    text = "SIMPLE"
    bbox = td.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    tx = (S - tw) // 2
    ty = int(S * 0.84)

    # Text shadow
    td.text((tx + 2, ty + 2), text, fill=(0, 0, 0, 60), font=font)
    # Main text
    td.text((tx, ty), text, fill=(255, 255, 255, 240), font=font)

    # "AI" subscript in accent color
    ai_size = int(S * 0.065)
    try:
        ai_font = ImageFont.truetype("segoeuib.ttf", ai_size)
    except:
        try:
            ai_font = ImageFont.truetype("arialbd.ttf", ai_size)
        except:
            ai_font = font

    ai_bbox = td.textbbox((0, 0), "AI", font=ai_font)
    ai_w = ai_bbox[2] - ai_bbox[0]
    ai_x = tx + tw + int(S * 0.015)
    ai_y = ty + int(S * 0.025)
    td.text((ai_x, ai_y), "AI", fill=(180, 140, 255, 255), font=ai_font)

    img = Image.alpha_composite(img, text_layer)

    return img


def create_ico(img, ico_path):
    """Save as multi-resolution ICO."""
    sizes = [256, 128, 64, 48, 32, 16]
    frames = []
    for s in sizes:
        resized = img.resize((s, s), Image.LANCZOS)
        frames.append(resized)
    # ICO: save the largest as base, include all sizes
    frames[0].save(ico_path, format="ICO", sizes=[(s, s) for s in sizes],
                   append_images=frames[1:])


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("Generating SIMPLE_AI logo...")
    logo = create_logo()

    png_path = os.path.join(out_dir, "simple_ai_logo.png")
    logo.save(png_path, "PNG")
    print(f"  PNG saved: {png_path}")

    ico_path = os.path.join(out_dir, "simple_ai.ico")
    create_ico(logo, ico_path)
    print(f"  ICO saved: {ico_path}")

    # Also save a smaller web favicon
    favicon = logo.resize((192, 192), Image.LANCZOS)
    fav_path = os.path.join(out_dir, "web", "favicon.png")
    if os.path.isdir(os.path.join(out_dir, "web")):
        favicon.save(fav_path, "PNG")
        print(f"  Web favicon: {fav_path}")

    print("Done!")
