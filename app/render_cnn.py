# app/render_cnn.py
from manim import *
import json
import tempfile
import subprocess
from pathlib import Path

MEDIA_DIR = Path("media/videos/CNNScene")
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


# ---------- 유틸: 레이어 박스/라벨 ----------
def make_layer_box(label: str, x: float, y: float = 0, color=BLUE) -> VGroup:
    box = Rectangle(width=2.2, height=1.1, color=color).shift(RIGHT * x + UP * y)
    text = Text(label, font_size=28).move_to(box.get_center())
    return VGroup(box, text)


# ---------- 각 레이어별 연출(가벼운 버전) ----------
class CNNAnimator:
    def __init__(self, scene: Scene):
        self.scene = scene

    def animate_input_blob(self, x: float = -5.0, y: float = 0.0) -> Mobject:
        blob = Square(side_length=0.8, color=YELLOW, fill_opacity=0.6).shift(RIGHT * x + UP * y)
        self.scene.play(FadeIn(blob))
        return blob

    def animate_move_to_layer(self, mobj: Mobject, layer_box: VGroup, dx: float = -0.7, run_time: float = 0.7):
        # 레이어 왼쪽 근처로 부드럽게 이동
        target = layer_box[0].get_left() + LEFT * 0.3
        self.scene.play(mobj.animate.move_to(target), run_time=run_time)

    def animate_conv(self, layer_box: VGroup):
        # 간단한 커널 슬라이딩 효과
        kernel = Square(0.4, color=RED, fill_opacity=0.4)
        kernel.move_to(layer_box[0].get_left() + RIGHT * 0.4)
        self.scene.play(FadeIn(kernel), run_time=0.2)
        for _ in range(3):
            self.scene.play(kernel.animate.shift(RIGHT * 0.4), run_time=0.25)
        for _ in range(3):
            self.scene.play(kernel.animate.shift(LEFT * 0.4), run_time=0.25)
        self.scene.play(FadeOut(kernel), run_time=0.2)

        # 결과 피처맵 미니 스택 (시각적 피드백)
        fm = VGroup(*[
            Rectangle(width=0.35, height=0.35, color=WHITE, fill_opacity=0.3)
            .shift(layer_box[0].get_right() + RIGHT * (0.25 * (i+1)))
            for i in range(3)
        ])
        self.scene.play(LaggedStart(*[FadeIn(r) for r in fm], lag_ratio=0.1), run_time=0.5)
        self.scene.play(LaggedStart(*[FadeOut(r) for r in fm], lag_ratio=0.08), run_time=0.4)

    def animate_relu(self, layer_box: VGroup):
        self.scene.play(Flash(layer_box[0], color=ORANGE), run_time=0.4)

    def animate_pool(self, layer_box: VGroup):
        pool = Rectangle(width=1.0, height=1.0, color=GREEN, fill_opacity=0.25)
        pool.move_to(layer_box[0].get_center())
        self.scene.play(FadeIn(pool), run_time=0.2)
        self.scene.play(pool.animate.scale(0.6), run_time=0.5)
        self.scene.play(FadeOut(pool), run_time=0.2)

    def animate_flatten(self, layer_box: VGroup):
        # 2D → 1D 느낌: 작은 정사각형들이 일렬로 재배열
        squares = VGroup(*[
            Square(0.18, color=YELLOW, fill_opacity=0.6)
            for _ in range(8)
        ]).arrange_in_grid(rows=2, cols=4, buff=0.06).move_to(layer_box[0].get_center())
        self.scene.play(FadeIn(squares), run_time=0.3)
        self.scene.play(squares.animate.arrange(RIGHT, buff=0.06).scale(0.9), run_time=0.6)
        self.scene.play(FadeOut(squares), run_time=0.2)

    def animate_dense(self, layer_box: VGroup):
        # 입력 벡터 → 출력 노드로 연결선 몇 개 (가벼운 연출)
        left_vec = VGroup(*[Dot(layer_box[0].get_left() + LEFT * (0.8 - 0.15 * i)) for i in range(4)])
        right_nodes = VGroup(*[
            Circle(0.08, color=BLUE, fill_opacity=0.5)
            .shift(layer_box[0].get_right() + RIGHT * 0.8 + DOWN * (0.25 * (i - 2)))
            for i in range(5)
        ])
        self.scene.play(LaggedStart(*[FadeIn(d) for d in left_vec], lag_ratio=0.1), run_time=0.4)
        self.scene.play(LaggedStart(*[FadeIn(n) for n in right_nodes], lag_ratio=0.08), run_time=0.5)

        edges = VGroup()
        for d in left_vec:
            for n in right_nodes:
                line = Line(d.get_center(), n.get_center(), stroke_opacity=0.4)
                edges.add(line)
        self.scene.play(LaggedStart(*[Create(e) for e in edges], lag_ratio=0.02), run_time=0.6)
        self.scene.wait(0.25)
        self.scene.play(FadeOut(left_vec), FadeOut(right_nodes), FadeOut(edges), run_time=0.3)


def render_cnn_scene(ir: dict, out_basename="cnn_demo", fmt="mp4") -> str:
    """
    IR(JSON)의 layers 정보를 이용해
    - 레이어 박스 VGroup 계층 구성
    - 입력 blob이 레이어를 통과하며 각 레이어별 기본 연출 수행
    으로 영상을 생성한다.
    """
    ir_json_str = json.dumps(ir, ensure_ascii=False, indent=2)

    # Manim Scene 템플릿
    scene_code = f"""
from manim import *
import json

class IRScene(Scene):
    def construct(self):
        IR = json.loads(r'''{ir_json_str}''')

        layers_cfg = IR.get("layers", [])
        layer_groups = []
        x = -3.5
        for cfg in layers_cfg:
            name = cfg.get("type", "Layer")
            color = BLUE
            if name.lower().startswith("conv"):
                color = YELLOW
            elif name.lower().startswith("relu"):
                color = ORANGE
            elif name.lower().startswith("maxpool") or name.lower().startswith("pool"):
                color = GREEN
            elif name.lower().startswith("flatten"):
                color = PURPLE
            elif name.lower().startswith("dense") or name.lower().startswith("linear"):
                color = TEAL

            box = Rectangle(width=2.2, height=1.1, color=color).shift(RIGHT * x)
            text = Text(name, font_size=28).move_to(box.get_center())
            grp = VGroup(box, text)
            self.add(grp)
            layer_groups.append(grp)
            x += 2.7

        # 입력 blob 등장
        blob = Square(side_length=0.8, color=YELLOW, fill_opacity=0.6).shift(LEFT*5.0)
        self.play(FadeIn(blob), run_time=1.0)
        self.wait(0.5)

        # --- 레이어별 연출 함수들 ---
        def conv_effect(layer):
            kernel = Square(0.4, color=RED, fill_opacity=0.4)
            kernel.move_to(layer[0].get_left() + RIGHT * 0.4)
            self.play(FadeIn(kernel), run_time=0.5)
            for _ in range(3):
                self.play(kernel.animate.shift(RIGHT * 0.4), run_time=0.5)
            for _ in range(3):
                self.play(kernel.animate.shift(LEFT * 0.4), run_time=0.5)
            self.play(FadeOut(kernel), run_time=0.4)

            fm = VGroup(*[
                Rectangle(width=0.35, height=0.35, color=WHITE, fill_opacity=0.3)
                .shift(layer[0].get_right() + RIGHT * (0.25 * (i+1)))
                for i in range(3)
            ])
            self.play(LaggedStart(*[FadeIn(r) for r in fm], lag_ratio=0.15), run_time=1.0)
            self.play(LaggedStart(*[FadeOut(r) for r in fm], lag_ratio=0.1), run_time=0.8)

        def relu_effect(layer):
            self.play(Flash(layer[0], color=ORANGE), run_time=0.7)

        def pool_effect(layer):
            pool = Rectangle(width=1.0, height=1.0, color=GREEN, fill_opacity=0.25)
            pool.move_to(layer[0].get_center())
            self.play(FadeIn(pool), run_time=0.5)
            self.play(pool.animate.scale(0.6), run_time=0.8)
            self.play(FadeOut(pool), run_time=0.5)

        def flatten_effect(layer):
            squares = VGroup(*[
                Square(0.18, color=YELLOW, fill_opacity=0.6)
                for _ in range(8)
            ]).arrange_in_grid(rows=2, cols=4, buff=0.06).move_to(layer[0].get_center())
            self.play(FadeIn(squares), run_time=0.5)
            self.play(squares.animate.arrange(RIGHT, buff=0.06).scale(0.9), run_time=0.8)
            self.play(FadeOut(squares), run_time=0.4)

        def dense_effect(layer):
            left_vec = VGroup(*[Dot(layer[0].get_left() + LEFT * (0.8 - 0.15 * i)) for i in range(4)])
            right_nodes = VGroup(*[
                Circle(0.08, color=BLUE, fill_opacity=0.5)
                .shift(layer[0].get_right() + RIGHT * 0.8 + DOWN * (0.25 * (i - 2)))
                for i in range(5)
            ])
            self.play(LaggedStart(*[FadeIn(d) for d in left_vec], lag_ratio=0.1), run_time=0.6)
            self.play(LaggedStart(*[FadeIn(n) for n in right_nodes], lag_ratio=0.08), run_time=0.8)

            edges = VGroup()
            for d in left_vec:
                for n in right_nodes:
                    line = Line(d.get_center(), n.get_center(), stroke_opacity=0.4)
                    edges.add(line)
            self.play(LaggedStart(*[Create(e) for e in edges], lag_ratio=0.03), run_time=1.0)
            self.wait(0.5)
            self.play(FadeOut(left_vec), FadeOut(right_nodes), FadeOut(edges), run_time=0.5)

        # --- 순차 실행 ---
        for cfg, layer in zip(layers_cfg, layer_groups):
            target = layer[0].get_left() + LEFT * 0.4
            self.play(blob.animate.move_to(target), run_time=0.8)
            self.wait(0.3)

            t = cfg.get("type", "").lower()
            if t.startswith("conv"):
                conv_effect(layer)
            elif t.startswith("relu"):
                relu_effect(layer)
            elif t.startswith("maxpool") or t.startswith("pool"):
                pool_effect(layer)
            elif t.startswith("flatten"):
                flatten_effect(layer)
            elif t.startswith("dense") or t.startswith("linear"):
                dense_effect(layer)
            else:
                self.wait(0.3)

        # 종료
        self.play(blob.animate.set_color(GREEN), run_time=1.0)
        self.play(FadeOut(blob), *[grp[0].animate.set_color(GREY) for grp in layer_groups], run_time=1.0)
        self.wait(2.0)  # ✅ 최종 프레임 유지

"""

    # 임시 파일에 작성 → manim 호출
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(scene_code)
        tmp_path = tmp.name

    cmd = ["manim", "-ql", tmp_path, "IRScene", "--format", fmt, "-o", f"{out_basename}.{fmt}"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Manim rendering failed: {e}")

    return str(MEDIA_DIR / f"{out_basename}.{fmt}")
