from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "poster" / "he_shu_zhi_nong_product_poster_illustrated.png"
OUTPUT = ROOT / "output" / "poster" / "he_shu_zhi_nong_product_poster_1200x560.png"
FONT = Path(r"C:\Windows\Fonts\msyh.ttc")


def font(size, bold=False):
    return ImageFont.truetype(str(FONT), size, index=1 if bold else 0)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    base = Image.open(SOURCE).convert("RGBA")
    # 保留插画中田野、温室与农业设施的主体，作为右侧视觉区域。
    scene = base.crop((0, 480, base.width, 1240)).resize((650, 560), Image.Resampling.LANCZOS)
    image = Image.new("RGBA", (1200, 560), (247, 240, 216, 255))
    image.paste(scene, (550, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, 660, 560), fill=(248, 242, 219, 255))
    draw.ellipse((-145, 420, 390, 760), fill=(223, 194, 102, 150))
    draw.ellipse((-120, 450, 510, 770), fill=(63, 122, 77, 225))
    draw.rounded_rectangle((44, 50, 565, 266), radius=24, fill=(255, 250, 234, 222))
    draw.text((76, 76), "禾枢智农", font=font(64, True), fill=(33, 91, 53, 255))
    draw.text((80, 151), "多智能体智慧种植决策与管控平台", font=font(25), fill=(51, 103, 67, 255))
    draw.rounded_rectangle((76, 202, 535, 240), radius=14, fill=(44, 112, 69, 235))
    draw.text((94, 209), "感知 · 分析 · 决策 · 执行 · 记录", font=font(18), fill=(255, 255, 246, 255))
    draw.rounded_rectangle((44, 470, 610, 520), radius=17, fill=(33, 91, 53, 232))
    draw.text((72, 484), "多智能体协同  |  视觉巡检  |  IoT 安全联动", font=font(18), fill=(255, 255, 245, 255))
    image.convert("RGB").save(OUTPUT, "PNG", optimize=True)
    print(OUTPUT)
    print(Image.open(OUTPUT).size)


if __name__ == "__main__":
    main()
