"""确定性地合成 MOSS-Image 官方格式的 layout 提示词。

公开语料里没有任何 MOSS schema 的 layout 文本（shuixian 的 JSON 是 Midjourney
自己的 schema，Goku 的是 gpt-image 指令式），而布局格式的 token（大括号、键名、
负数坐标、转义换行）在纯散文语料里几乎不出现。所以这一桶必须合成。

合成不调用任何模型，只用固定模板 + 固定词表 + 固定随机种子，因此同样参数
永远产出同一份数据，别人可以复现。

序列化规则严格对齐生成侧（ComfyUI 的 MOSS-Image Layout Prompt 节点）：

    {"summary":{...},"items":[...],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}

- 紧凑单行，`separators=(",", ":")`，`ensure_ascii=False`
- `bbox` 是整数 `[y_min, x_min, y_max, x_max]`，以画布中心为原点
- 草稿坐标是相对左上角的 `[x1, y1, x2, y2]` 像素；转换时左/上取 floor、
  右/下取 ceil，再减去画布中心
"""

from __future__ import annotations

import json
import math
import random
from typing import Iterator, Sequence

# 画布边长必须是 16 的正整数倍，且覆盖横竖方三种比例：
# bbox 是绝对像素，画布不变坐标数字就不变，尺寸多样性直接决定坐标 token 的多样性。
CANVASES: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (1536, 1024),
    (1024, 1536),
    (2048, 512),
    (512, 2048),
    (1344, 768),
    (768, 1344),
    (1152, 896),
    (896, 1152),
    (1216, 832),
)

LABELS = ("object", "object", "object", "text", "text", "character")
# 官方 assets 里真实出现过 panel / subject，这两个 label 不在节点产出的集合里，
# 但既然真实数据里有，就少量掺一点，避免投影对它们完全没有覆盖。
RARE_LABELS = ("panel", "subject")

HEX_COLORS = (
    "#D4AF37", "#FFFFFF", "#B22222", "#87CEEB", "#2F4F4F", "#F5F0E5", "#292824",
    "#496878", "#FFF7E8", "#E8B64C", "#202D45", "#BA3034", "#D6A080", "#F2F0EA",
)

SUMMARY_SCENES_EN = (
    "A minimalist travel poster with a cream background and a flat blue mountain range.",
    "A warm studio campaign for a handmade ceramic cup, with an ivory backdrop and soft side lighting.",
    "A cinematic 16:9 gameplay screenshot from a dark-fantasy action game.",
    "A high-quality indoor photograph of three trainers guiding a client through a weighted squat.",
    "A finished Chinese university entrance examination practice paper rendered as a flat scan.",
    "An amateur selfie contact sheet presented as a coherent three-by-three grid.",
    "A retro-futuristic suburban street scene with brutalist architecture and neon signage.",
    "A children's picture book spread with warm gouache textures and handwritten lettering.",
    "A luxury skincare advertisement layout with glossy packaging and cool blue lighting.",
    "A vertical infographic about urban beekeeping with a muted sage palette.",
)

SUMMARY_SCENES_ZH = (
    "极简旅行海报，米色背景，蓝色山脉插图，标题位于顶部居中。",
    "手工陶瓷杯的暖色影棚广告，象牙白背景，柔和侧光，深绿色排版。",
    "中文语文教材页面排版，简洁排版，中国传统美学，柔和米色背景。",
    "健身房产品手册内页，冷灰色调，人物三视图与器械特写并置。",
    "手账风格的周末计划页，牛皮纸底色，彩色贴纸与手写批注。",
    "九宫格动漫角色立绘合集，每格独立构图，统一描边风格。",
    "川西秋季风景摄影集封面，晨雾中的彩林与远处雪山。",
    "咖啡馆季节限定菜单板，粉笔手写字体，木质纹理背景。",
    "心理学知识图解海报，蓝色系，三段式信息层级。",
    "国潮风月饼礼盒包装展开图，烫金纹样与红色底纹。",
)

CAPTION_OBJECT_EN = (
    "An ivory ceramic cup with a rounded handle and fine speckled glaze, shot at a three-quarter angle.",
    "A flat blue mountain silhouette with crisp edges and a soft drop shadow.",
    "A lone figure in a raincoat standing at the far end of a wooden pier.",
    "A chrome espresso machine with a brushed finish and visible steam.",
    "A weathered notice board leaning against a gravel shoulder.",
    "A cluster of autumn maple leaves resting on wet stone.",
    "A compact mechanical keyboard with warm white backlighting.",
    "A glass terrarium containing moss, small ferns and a ceramic figurine.",
    "A vintage bicycle parked beside a whitewashed wall.",
    "A stack of hardcover books with cloth spines and gilt lettering.",
)

CAPTION_OBJECT_ZH = (
    "一只带圆润把手的象牙白陶瓷杯，釉面有细密斑点，四分之三角度拍摄。",
    "一座边缘干净的蓝色山形剪影，带柔和投影。",
    "一个穿雨衣的身影站在木栈道尽头，画面留出大片天空。",
    "一台拉丝不锈钢外壳的半自动咖啡机，能看到蒸汽。",
    "一块靠在碎石路肩上的旧告示牌，铁皮有折痕与锈迹。",
    "几片枫叶落在湿润的石板上，叶脉清晰。",
    "一把暖白背光的紧凑机械键盘，键帽为浅灰配色。",
    "一个玻璃生态缸，里面有苔藓、小型蕨类和陶瓷小摆件。",
    "一辆停在白墙旁的复古自行车，车筐里有花。",
    "一摞布面精装书，书脊有烫金字。",
)

CAPTION_TEXT_EN = (
    "Large dark-blue uppercase title centered at the top, wide tracking.",
    "Small supporting copy in muted grey, comfortably legible on two lines.",
    "Bold black Song-style heading with natural square proportions.",
    "Handwritten annotation in dark ink with a slight forward slant.",
    "Compact sans-serif label with generous letter spacing.",
)

CAPTION_TEXT_ZH = (
    "顶部居中的深蓝色大号标题，字距较宽。",
    "灰调小号辅助文案，两行排布，清晰可读。",
    "黑色宋体粗标题，字形方正自然。",
    "深色手写批注，略带前倾。",
    "紧凑无衬线标签，字距舒展。",
)

TEXT_CONTENT = (
    "EXPLORE",
    "SALE 50%",
    "把日常，过成诗。",
    "一杯温暖\n一刻从容",
    "出师表",
    "MOSS-Image",
    "第 3 章 · 春分",
    "NEW ARRIVAL",
    "扫码了解更多",
    "OPEN 09:00 - 21:00",
    "记得买牛奶和鸡蛋！",
    "雾凇\n晴 -3°C",
    "Chapter One",
    "限时特惠\n仅此三天",
    "STAY CURIOUS",
)

SUMMARY_META = (
    "poster", "product campaign", "studio photography", "screenshot", "infographic",
    "editorial", "comic", "packaging", "menu", "photograph",
)

CANVAS_KINDS = ("single", "single", "single", "multi_panel", "collage")


def centered_box(
    box: Sequence[float], width: int, height: int
) -> list[int]:
    """把左上角原点的 xyxy 像素框转换成 MOSS 的中心原点 yxyx 整数框。

    Args:
        box: `[x1, y1, x2, y2]`，相对画布左上角，允许浮点。
        width: 画布宽，必须是 16 的正整数倍。
        height: 画布高，必须是 16 的正整数倍。

    Returns:
        `[y_min, x_min, y_max, x_max]`，整数，以画布中心为原点，x 向右、y 向下。

    Raises:
        ValueError: 框越界、宽高非正、或 floor/ceil 后宽高退化为 0。
    """
    if width <= 0 or height <= 0 or width % 16 or height % 16:
        raise ValueError(f"画布必须是 16 的正整数倍，收到 {width}x{height}")
    if len(box) != 4:
        raise ValueError(f"bbox 必须是四个数，收到 {box!r}")
    x1, y1, x2, y2 = (float(v) for v in box)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError(f"bbox 含非有限数：{box!r}")
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"bbox 越界或宽高非正：{box!r} 画布 {width}x{height}")

    # 先用像素边包住再居中：floor 左/上、ceil 右/下，与生成侧一致。
    y_min = math.floor(round(y1, 9)) - height // 2
    x_min = math.floor(round(x1, 9)) - width // 2
    y_max = math.ceil(round(y2, 9)) - height // 2
    x_max = math.ceil(round(x2, 9)) - width // 2
    if not (y_min < y_max and x_min < x_max):
        raise ValueError(f"框在像素精度下退化：{box!r}")
    return [y_min, x_min, y_max, x_max]


def compile_draft(draft: dict) -> str:
    """把可读的 layout 草稿编译成模型看的单行 JSON。

    Args:
        draft: 含 `width` `height` `bbox_coord_space="abs"` `bbox_format="xyxy"`
            `summary` 与 `items` 的字典。`items` 每项至少含 `label` `caption` `bbox`；
            `text` 项的 `text` 为要渲染的字面文字；可选 `palette` `tags`。

    Returns:
        紧凑 JSON 字符串，键顺序为 `summary` → `items` → `bbox_coord_space`
        → `bbox_format`，`items` 内保持草稿给定的键顺序（`bbox` 原位替换成整数）。

    Raises:
        ValueError: 坐标非法、字段缺失或出现未知字段。

    Notes:
        键顺序会影响 token 序列，所以不能换成 `sort_keys=True`。
    """
    width, height = int(draft["width"]), int(draft["height"])
    if draft.get("bbox_coord_space") != "abs":
        raise ValueError("草稿的 bbox_coord_space 必须是 abs")
    if draft.get("bbox_format") != "xyxy":
        raise ValueError("草稿的 bbox_format 必须是 xyxy")

    summary = draft["summary"]
    allowed_summary = {"caption", "meta", "canvas", "palette"}
    unknown = set(summary) - allowed_summary
    if unknown:
        raise ValueError(f"summary 出现未知字段：{sorted(unknown)}")

    compiled_items = []
    for index, item in enumerate(draft["items"]):
        allowed_item = {"label", "caption", "bbox", "text", "tags", "palette"}
        unknown = set(item) - allowed_item
        if unknown:
            raise ValueError(f"items[{index}] 出现未知字段：{sorted(unknown)}")
        missing = {"label", "caption", "bbox"} - set(item)
        if missing:
            raise ValueError(f"items[{index}] 缺少字段：{sorted(missing)}")
        converted = {}
        for key, value in item.items():
            converted[key] = (
                centered_box(value, width, height) if key == "bbox" else value
            )
        compiled_items.append(converted)

    result = {
        "summary": summary,
        "items": compiled_items,
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def split_canvas(
    width: int, height: int, parts: int, rng: random.Random, min_side: int = 72
) -> list[tuple[float, float, float, float]]:
    """把画布递归二分成若干互不重叠的矩形，用于生成真实的布局框。

    Args:
        width: 画布宽。
        height: 画布高。
        parts: 期望的矩形数量，>= 1。
        rng: 随机源，决定切分方向与切分比例。
        min_side: 任一边小于该值就不再继续切。

    Returns:
        `[(x1, y1, x2, y2), ...]` 浮点框列表，长度等于 `parts`（受 `min_side` 限制）。

    Notes:
        用 guillotine 切分而不是随机撒框，保证框之间不重叠、不留离谱缝隙，
        坐标分布和真实排版接近。
    """
    if parts < 1:
        raise ValueError("parts 必须 >= 1")
    boxes: list[tuple[float, float, float, float]] = [(0.0, 0.0, float(width), float(height))]
    while len(boxes) < parts:
        # 每次切当前面积最大的框，避免出现一条细长碎片。
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        box = boxes.pop(0)
        w, h = box[2] - box[0], box[3] - box[1]
        horizontal = w >= h if w != h else rng.random() < 0.5
        if horizontal and w < 2 * min_side:
            horizontal = False
        if not horizontal and h < 2 * min_side:
            horizontal = True
        if horizontal and w < 2 * min_side:
            boxes.append(box)
            break
        if not horizontal and h < 2 * min_side:
            boxes.append(box)
            break
        ratio = rng.uniform(0.38, 0.62)
        if horizontal:
            cut = box[0] + w * ratio
            boxes.extend([(box[0], box[1], cut, box[3]), (cut, box[1], box[2], box[3])])
        else:
            cut = box[1] + h * ratio
            boxes.extend([(box[0], box[1], box[2], cut), (box[0], cut, box[2], box[3])])
    return boxes


def _pick(rng: random.Random, pool: Sequence[str]) -> str:
    return pool[rng.randrange(len(pool))]


def _build_draft(rng: random.Random, lang: str, variant: str) -> dict:
    """按语言与序列化风格拼一份 layout 草稿。

    Args:
        rng: 已按样本索引播种的随机源。
        lang: `"zh"` 或 `"en"`，决定 caption 与文字内容用哪种语言。
        variant: `"comfy"` 对齐 ComfyUI 节点的键顺序与字段集合；
            `"skill"` 对齐 compile_prompt.py 允许的额外字段（meta/canvas/tags）。

    Returns:
        可直接传给 `compile_draft` 的草稿字典。
    """
    width, height = CANVASES[rng.randrange(len(CANVASES))]
    n_items = rng.randint(2, 6)
    boxes = split_canvas(width, height, n_items, rng)

    if lang == "zh":
        summary_caption = _pick(rng, SUMMARY_SCENES_ZH)
        obj_pool, text_pool = CAPTION_OBJECT_ZH, CAPTION_TEXT_ZH
    else:
        summary_caption = _pick(rng, SUMMARY_SCENES_EN)
        obj_pool, text_pool = CAPTION_OBJECT_EN, CAPTION_TEXT_EN

    # 少量样本复刻官方 assets 里出现的空 summary（002 号就是这样）。
    if variant == "skill" and rng.random() < 0.05:
        summary: dict = {}
    elif variant == "skill":
        summary = {"caption": summary_caption}
        if rng.random() < 0.6:
            summary["meta"] = [_pick(rng, SUMMARY_META)]
        if rng.random() < 0.5:
            summary["canvas"] = _pick(rng, CANVAS_KINDS)
        if rng.random() < 0.25:
            summary["palette"] = rng.sample(HEX_COLORS, rng.randint(2, 4))
    else:
        summary = {"caption": summary_caption}

    items = []
    for box in boxes:
        label = _pick(rng, LABELS)
        if variant == "skill" and rng.random() < 0.08:
            label = _pick(rng, RARE_LABELS)
        if label == "text":
            item = {"label": "text", "caption": _pick(rng, text_pool)}
            item["text"] = _pick(rng, TEXT_CONTENT)
            if variant == "skill" and rng.random() < 0.3:
                item["tags"] = ["typography", "lettering"]
            if rng.random() < 0.35:
                item["palette"] = rng.sample(HEX_COLORS, rng.randint(2, 3))
        else:
            item = {"label": label, "caption": _pick(rng, obj_pool)}
            if variant == "skill" and rng.random() < 0.5:
                item["tags"] = [rng.choice(("subject", "detail", "background", "prop"))]
            if rng.random() < 0.35:
                item["palette"] = rng.sample(HEX_COLORS, rng.randint(2, 4))
        item["bbox"] = [round(v, 3) for v in box]
        items.append(item)

    return {
        "width": width,
        "height": height,
        "bbox_coord_space": "abs",
        "bbox_format": "xyxy",
        "summary": summary,
        "items": items,
    }


def generate_layouts(
    count: int, seed: int, lang_mix: float = 0.5
) -> Iterator[dict]:
    """批量生成 layout 提示词。

    Args:
        count: 生成条数。
        seed: 全局随机种子；每条样本的随机源由 `seed*100003 + index` 决定，
            所以增删条数不会改变已有样本的内容。
        lang_mix: 中文样本占比，0~1。

    Yields:
        `{"text": 编译后的提示词, "lang": "zh"/"en", "variant": "comfy"/"skill",
        "width": int, "height": int, "n_items": int}`。

    Notes:
        每条样本独立播种，保证「先跑 200 条看看」和「跑满 1000 条」前 200 条完全一致。
    """
    for index in range(count):
        rng = random.Random(seed * 100_003 + index)
        lang = "zh" if rng.random() < lang_mix else "en"
        variant = "comfy" if rng.random() < 0.8 else "skill"
        draft = _build_draft(rng, lang, variant)
        yield {
            "text": compile_draft(draft),
            "lang": lang,
            "variant": variant,
            "width": draft["width"],
            "height": draft["height"],
            "n_items": len(draft["items"]),
        }
