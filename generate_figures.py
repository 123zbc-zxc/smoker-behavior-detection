"""
毕业设计科研图生成脚本
生成论文前三章所需全部示意图
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = r"D:\Smoker Behavior Detection Based on Deep Learning\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 图1: 目标检测方法分类树状图
# ============================================================
def fig1_detection_tree():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    def draw_box(ax, x, y, w, h, text, fc='#4472C4', tc='white', fontsize=9, radius=0.25):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle=f"round,pad=0.05,rounding_size={radius}",
                              linewidth=1.2, edgecolor='#2E4057',
                              facecolor=fc, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color=tc, fontweight='bold', zorder=4,
                wrap=True)

    def draw_line(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-', color='#555555',
                                   lw=1.5), zorder=2)

    # 根节点
    draw_box(ax, 8, 8.2, 2.8, 0.7, '目标检测方法', fc='#1F3864', fontsize=11)

    # 二级节点
    draw_box(ax, 4, 6.8, 3.0, 0.65, '传统目标检测方法', fc='#2E75B6', fontsize=10)
    draw_box(ax, 12, 6.8, 3.2, 0.65, '基于深度学习的检测方法', fc='#2E75B6', fontsize=10)

    draw_line(ax, 8, 7.85, 4, 7.12)
    draw_line(ax, 8, 7.85, 12, 7.12)

    # 传统方法的子节点
    trad_children = [
        (2.2, 5.5, '模板匹配方法', '#5B9BD5'),
        (4.0, 5.5, '图像分析方法', '#5B9BD5'),
        (5.8, 5.5, '传统机器学习\n方法', '#5B9BD5'),
    ]
    for cx, cy, ct, cf in trad_children:
        draw_box(ax, cx, cy, 1.55, 0.65, ct, fc=cf, fontsize=8.5)
        draw_line(ax, 4, 6.47, cx, cy + 0.32)

    # 传统机器学习细分
    ml_children = [
        (4.8, 4.15, 'HOG+SVM', '#BDD7EE'),
        (6.5, 4.15, 'Haar+\nAdaBoost', '#BDD7EE'),
    ]
    for cx, cy, ct, cf in ml_children:
        draw_box(ax, cx, cy, 1.4, 0.6, ct, fc=cf, tc='#1F3864', fontsize=8)
        draw_line(ax, 5.8, 5.17, cx, cy + 0.3)

    # 深度学习方法的子节点
    draw_box(ax, 10.5, 5.5, 2.5, 0.65, '有锚框检测方法\n(Anchor-Based)', fc='#5B9BD5', fontsize=8.5)
    draw_box(ax, 13.5, 5.5, 2.5, 0.65, '无锚框检测方法\n(Anchor-Free)', fc='#5B9BD5', fontsize=8.5)
    draw_line(ax, 12, 6.47, 10.5, 5.82)
    draw_line(ax, 12, 6.47, 13.5, 5.82)

    # 有锚框细分
    ab_children = [
        (9.3, 4.15, '单阶段\n(YOLO系列,SSD)', '#BDD7EE'),
        (11.7, 4.15, '双阶段\n(Faster R-CNN)', '#BDD7EE'),
    ]
    for cx, cy, ct, cf in ab_children:
        draw_box(ax, cx, cy, 2.1, 0.65, ct, fc=cf, tc='#1F3864', fontsize=8)
        draw_line(ax, 10.5, 5.17, cx, cy + 0.32)

    # YOLO细分
    yolo_children = [
        (8.2, 2.8, 'YOLOv5', '#DEEAF1'),
        (9.3, 2.8, 'YOLOv8', '#DEEAF1'),
        (10.4, 2.8, 'YOLO11', '#DEEAF1'),
    ]
    for cx, cy, ct, cf in yolo_children:
        draw_box(ax, cx, cy, 0.95, 0.5, ct, fc=cf, tc='#1F3864', fontsize=7.5)
        draw_line(ax, 9.3, 3.82, cx, cy + 0.25)

    # 无锚框细分
    af_children = [
        (12.9, 4.15, '基于关键点\n(CenterNet)', '#BDD7EE'),
        (14.8, 4.15, '基于边缘\n(FCOS)', '#BDD7EE'),
    ]
    for cx, cy, ct, cf in af_children:
        draw_box(ax, cx, cy, 1.7, 0.65, ct, fc=cf, tc='#1F3864', fontsize=8)
        draw_line(ax, 13.5, 5.17, cx, cy + 0.32)

    ax.set_title('图1  目标检测方法分类', fontsize=13, fontweight='bold', pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图1_目标检测方法分类树状图.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图1 完成")


# ============================================================
# 图2: 传统目标检测算法流程图
# ============================================================
def fig2_traditional_flowchart():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')

    def draw_rect(ax, x, y, w, h, text, fc='#4472C4', tc='white', fs=9):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.05,rounding_size=0.15",
                              linewidth=1.5, edgecolor='#1F3864', facecolor=fc, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fs, color=tc, fontweight='bold', zorder=4)

    def draw_arrow(ax, x1, x2, y=2.0):
        ax.annotate('', xy=(x2 - 0.05, y), xytext=(x1 + 0.05, y),
                    arrowprops=dict(arrowstyle='->', color='#2E4057',
                                   lw=2.0, mutation_scale=18), zorder=2)

    steps = [
        (1.2, '输入\n图像', '#1F3864'),
        (3.5, '候选区域生成\n(滑动窗口/选择性搜索)', '#2E75B6'),
        (6.5, '特征提取\n(SIFT / HOG / Haar)', '#2E75B6'),
        (9.5, '分类器\n(SVM / AdaBoost)', '#2E75B6'),
        (12.5, '输出\n检测结果', '#1F3864'),
    ]
    widths = [1.6, 2.8, 2.8, 2.5, 1.8]

    for i, ((x, text, fc), w) in enumerate(zip(steps, widths)):
        draw_rect(ax, x, 2.0, w, 1.1, text, fc=fc)
        if i < len(steps) - 1:
            draw_arrow(ax, x + w / 2, steps[i + 1][0] - widths[i + 1] / 2)

    # 添加子说明
    ax.text(3.5, 0.75, '滑动窗口：多尺度遍历\n选择性搜索：颜色/纹理聚合', ha='center',
            fontsize=7.5, color='#444444', style='italic')
    ax.text(6.5, 0.75, '手工设计特征描述符\n捕获形状/梯度/纹理信息', ha='center',
            fontsize=7.5, color='#444444', style='italic')
    ax.text(9.5, 0.75, '训练好的有监督分类器\n输出类别概率与边界框', ha='center',
            fontsize=7.5, color='#444444', style='italic')

    ax.set_title('图2  传统目标检测算法流程图', fontsize=13, fontweight='bold', pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图2_传统目标检测算法流程图.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图2 完成")


# ============================================================
# 图3: 卷积神经网络结构示意图
# ============================================================
def fig3_cnn_structure():
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5)
    ax.axis('off')

    def draw_layer(ax, x, y_center, width, height, n_layers, color, label, sublabel=''):
        gap = 0.12
        total = n_layers * height + (n_layers - 1) * gap
        y_start = y_center - total / 2
        for i in range(n_layers):
            rect = FancyBboxPatch((x, y_start + i * (height + gap)), width, height,
                                   boxstyle="square,pad=0.0",
                                   linewidth=1.0, edgecolor='white',
                                   facecolor=color, alpha=0.85, zorder=3)
            ax.add_patch(rect)
        ax.text(x + width / 2, y_center - total / 2 - 0.35, label,
                ha='center', va='top', fontsize=9, fontweight='bold', color='#1F3864')
        if sublabel:
            ax.text(x + width / 2, y_center - total / 2 - 0.65, sublabel,
                    ha='center', va='top', fontsize=7.5, color='#555555')

    def draw_conn_arrow(ax, x1, x2, y=2.5):
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color='#888888',
                                   lw=1.5, mutation_scale=14), zorder=2)

    # 输入图像
    ax.imshow(np.random.rand(28, 28, 3) * 0.3 + 0.35,
              extent=[0.3, 1.5, 1.2, 3.8], aspect='auto', zorder=3)
    ax.text(0.9, 0.75, '输入层\n(H×W×3)', ha='center', fontsize=8.5,
            fontweight='bold', color='#1F3864')

    # 卷积层1
    draw_layer(ax, 2.0, 2.5, 0.55, 0.6, 6, '#4472C4', '卷积层1\nConv', '32 filters, 3×3')
    draw_conn_arrow(ax, 1.5, 2.0, 2.5)

    # 池化层1
    draw_layer(ax, 3.3, 2.5, 0.45, 0.52, 6, '#ED7D31', '池化层1\nPool', 'MaxPool 2×2')
    draw_conn_arrow(ax, 2.55, 3.3, 2.5)

    # 卷积层2
    draw_layer(ax, 4.5, 2.5, 0.42, 0.45, 8, '#4472C4', '卷积层2\nConv', '64 filters, 3×3')
    draw_conn_arrow(ax, 3.75, 4.5, 2.5)

    # 池化层2
    draw_layer(ax, 5.65, 2.5, 0.35, 0.38, 8, '#ED7D31', '池化层2\nPool', 'MaxPool 2×2')
    draw_conn_arrow(ax, 4.92, 5.65, 2.5)

    # 卷积层3
    draw_layer(ax, 6.7, 2.5, 0.32, 0.32, 10, '#4472C4', '卷积层3\nConv', '128 filters')
    draw_conn_arrow(ax, 6.0, 6.7, 2.5)

    # 展平
    draw_conn_arrow(ax, 7.02, 7.6, 2.5)
    ax.text(7.35, 2.9, 'Flatten', ha='center', fontsize=7.5, color='#777777', style='italic')

    # 全连接层
    n_fc = [8, 6, 4]
    colors_fc = ['#A9D18E', '#70AD47', '#375623']
    labels_fc = ['全连接层\nFC-512', '全连接层\nFC-256', '输出层\nSoftmax']
    x_fc = [8.3, 9.7, 11.1]

    for i, (xf, nf, cf, lf) in enumerate(zip(x_fc, n_fc, colors_fc, labels_fc)):
        spacing = 2.4 / (nf - 1)
        for j in range(nf):
            cy = 2.5 - 1.2 + j * spacing
            circ = plt.Circle((xf, cy), 0.15, color=cf, zorder=3, linewidth=0.8,
                               ec='white')
            ax.add_patch(circ)
        ax.text(xf, 0.75, lf, ha='center', fontsize=8.5,
                fontweight='bold', color='#1F3864')
        if i > 0:
            draw_conn_arrow(ax, x_fc[i - 1] + 0.15, xf - 0.15, 2.5)

    # 激活函数标注
    ax.text(8.3, 4.35, 'ReLU', ha='center', fontsize=7.5, color='#C00000',
            style='italic')
    ax.text(9.7, 4.35, 'ReLU', ha='center', fontsize=7.5, color='#C00000',
            style='italic')
    ax.text(11.1, 4.35, 'Softmax', ha='center', fontsize=7.5, color='#C00000',
            style='italic')

    # 输出标注
    ax.text(12.2, 2.5, '→ 类别概率分布\n(目标类别输出)',
            ha='left', va='center', fontsize=8.5, color='#1F3864',
            bbox=dict(boxstyle='round,pad=0.3', fc='#F2F2F2', ec='#AAAAAA'))

    ax.set_title('图3  卷积神经网络结构示意图', fontsize=13, fontweight='bold', pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图3_卷积神经网络结构示意图.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图3 完成")


# ============================================================
# 图4: 卷积操作示意图
# ============================================================
def fig4_convolution():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    input_data = [
        [1, 0, 1, 0, 0],
        [0, 2, 0, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 1, 1],
    ]
    kernel = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    output = [[4, 2, 3], [2, 3, 2], [3, 2, 4]]

    cell = 0.65

    def draw_grid(ax, data, x0, y0, highlight=None, grid_color='#4472C4',
                  cell_color='#DDEEFF', hl_color='#FFD966', label=''):
        rows = len(data)
        cols = len(data[0])
        for r in range(rows):
            for c in range(cols):
                hl = (highlight is not None and
                      r in range(highlight[0], highlight[0] + 3) and
                      c in range(highlight[1], highlight[1] + 3))
                fc = hl_color if hl else cell_color
                rect = plt.Rectangle((x0 + c * cell, y0 + (rows - 1 - r) * cell),
                                      cell, cell,
                                      linewidth=1.2, edgecolor=grid_color,
                                      facecolor=fc, zorder=3)
                ax.add_patch(rect)
                ax.text(x0 + c * cell + cell / 2,
                        y0 + (rows - 1 - r) * cell + cell / 2,
                        str(data[r][c]),
                        ha='center', va='center',
                        fontsize=10, fontweight='bold', color='#1F3864', zorder=4)
        if label:
            ax.text(x0 + cols * cell / 2, y0 - 0.35, label,
                    ha='center', fontsize=9, fontweight='bold', color='#1F3864')

    # 输入矩阵 (5×5)，高亮左上3×3
    draw_grid(ax, input_data, 0.5, 0.8, highlight=(0, 0),
              grid_color='#2E75B6', cell_color='#DDEEFF', label='输入特征图 (5×5)')

    # ×符号
    ax.text(4.2, 2.6, '⊛', ha='center', va='center', fontsize=22, color='#555555')

    # 卷积核 (3×3)
    draw_grid(ax, kernel, 4.7, 1.45, grid_color='#C55A11',
              cell_color='#FFE5CC', label='卷积核 (3×3, stride=1)')

    # =符号
    ax.text(7.0, 2.6, '=', ha='center', va='center', fontsize=24, color='#555555')

    # 输出矩阵 (3×3)
    draw_grid(ax, output, 7.5, 1.45, grid_color='#375623',
              cell_color='#E2EFDA', hl_color='#A9D18E', label='输出特征图 (3×3)')

    # 计算公式说明
    ax.text(6.0, 4.6, '输出值 = Σ(输入区域 × 卷积核对应元素)',
            ha='center', fontsize=9, color='#444444',
            bbox=dict(boxstyle='round,pad=0.35', fc='#F9F9F9', ec='#AAAAAA'))
    ax.text(6.0, 4.1, '左上角示例: 1×1+0×0+1×1+0×0+2×1+0×0+1×1+1×0+0×1 = 4',
            ha='center', fontsize=8.5, color='#C00000')

    # 箭头指示卷积核滑动方向
    ax.annotate('', xy=(3.25, 3.75), xytext=(2.6, 3.75),
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.0, mutation_scale=16))
    ax.text(2.9, 3.95, '滑动方向', ha='center', fontsize=7.5, color='#E74C3C')

    ax.set_title('图4  卷积操作示意图', fontsize=13, fontweight='bold', pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图4_卷积操作示意图.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图4 完成")


# ============================================================
# 图5: 全连接层结构示意图
# ============================================================
def fig5_fully_connected():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    layer_configs = [
        (1.5, 5, '#4472C4', 'X层\n(输入层)\n5个神经元'),
        (5.0, 4, '#ED7D31', 'Y层\n(隐藏层)\n4个神经元'),
        (8.5, 3, '#70AD47', 'Z层\n(输出层)\n3个神经元'),
    ]

    positions = []
    for (x, n, color, label) in layer_configs:
        ys = np.linspace(1.0, 5.0, n)
        layer_pos = []
        for y in ys:
            circ = plt.Circle((x, y), 0.28, color=color, zorder=4,
                               linewidth=1.5, ec='white')
            ax.add_patch(circ)
            layer_pos.append((x, y))
        positions.append(layer_pos)
        ax.text(x, 0.35, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color='#1F3864')

    # 连接线
    connection_colors = ['#AAAAAA', '#BBBBBB']
    for li in range(len(positions) - 1):
        for (x1, y1) in positions[li]:
            for (x2, y2) in positions[li + 1]:
                ax.plot([x1 + 0.28, x2 - 0.28], [y1, y2],
                        '-', color='#AAAACC', linewidth=0.7, alpha=0.7, zorder=2)

    # 层间标注
    ax.text(3.25, 5.5, 'W₁ (权重矩阵)', ha='center', fontsize=8.5,
            color='#C00000', style='italic')
    ax.text(6.75, 5.5, 'W₂ (权重矩阵)', ha='center', fontsize=8.5,
            color='#C00000', style='italic')

    ax.annotate('', xy=(3.8, 5.35), xytext=(2.0, 5.35),
                arrowprops=dict(arrowstyle='->', color='#C00000', lw=1.5))
    ax.annotate('', xy=(7.3, 5.35), xytext=(5.5, 5.35),
                arrowprops=dict(arrowstyle='->', color='#C00000', lw=1.5))

    ax.text(5.0, 5.78, 'Y = σ(W₁·X + b₁)', ha='center', fontsize=9,
            color='#1F3864', bbox=dict(boxstyle='round,pad=0.3',
                                       fc='#EBF3FB', ec='#4472C4'))
    ax.text(5.0, 5.78 - 0.55, 'Z = σ(W₂·Y + b₂)', ha='center', fontsize=9,
            color='#1F3864', bbox=dict(boxstyle='round,pad=0.3',
                                       fc='#FFF2E2', ec='#ED7D31'))

    ax.set_title('图5  全连接层结构示意图', fontsize=13, fontweight='bold', pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图5_全连接层结构示意图.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图5 完成")


# ============================================================
# 图6: YOLOv8各版本性能对比表（图形化表格）
# ============================================================
def fig6_yolov8_comparison():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis('off')

    columns = ['模型', '输入尺寸\n(像素)', 'mAP val\n50-95', 'CPU ONNX速度\n(ms/帧)',
               'A100 TensorRT速度\n(ms/帧)', '参数量\n(M)', 'FLOPs\n(B)']
    data = [
        ['YOLOv8n', '640', '37.3', '80.4', '0.99', '3.2', '8.7'],
        ['YOLOv8s', '640', '44.9', '128.4', '1.20', '11.2', '28.6'],
        ['YOLOv8m', '640', '50.2', '234.7', '1.83', '25.9', '78.9'],
        ['YOLOv8l', '640', '52.9', '375.2', '2.39', '43.7', '165.2'],
        ['YOLOv8x', '640', '53.9', '479.1', '3.53', '68.2', '257.8'],
    ]

    row_colors_header = ['#1F3864'] * len(columns)
    row_colors = [
        ['#DDEEFF'] * len(columns),
        ['#EEF4FF'] * len(columns),
        ['#DDEEFF'] * len(columns),
        ['#EEF4FF'] * len(columns),
        ['#DDEEFF'] * len(columns),
    ]
    # 高亮YOLOv8n行（本设计所选模型）
    row_colors[0] = ['#FFD966'] * len(columns)

    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0.05, 1, 0.95]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor('#1F3864')
        cell.set_text_props(color='white', fontweight='bold', fontsize=9.5)
        cell.set_height(0.22)

    for i in range(len(data)):
        for j in range(len(columns)):
            cell = table[i + 1, j]
            cell.set_facecolor(row_colors[i][j])
            cell.set_text_props(color='#1F3864', fontsize=10)
            if i == 0:
                cell.set_text_props(color='#7B3F00', fontweight='bold', fontsize=10)
            cell.set_height(0.14)

    # 标注所选模型
    ax.text(0.5, 0.01, '★ 黄色高亮行（YOLOv8n）为本设计所选基础模型',
            ha='center', fontsize=9, color='#7B3F00', style='italic',
            transform=ax.transAxes)

    ax.set_title('图6  YOLOv8各版本在COCO数据集上的性能对比',
                 fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图6_YOLOv8各版本性能对比表.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图6 完成")


# ============================================================
# 图7: YOLOv8n网络结构图
# ============================================================
def fig7_yolov8n_structure():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    def block(ax, x, y, w, h, text, fc='#4472C4', tc='white', fs=8.5, lw=1.2):
        b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                            boxstyle="round,pad=0.05,rounding_size=0.15",
                            linewidth=lw, edgecolor='#1F3864', facecolor=fc, zorder=3)
        ax.add_patch(b)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fs, color=tc, fontweight='bold', zorder=4)

    def arrow_v(ax, x, y1, y2, lw=2.0):
        ax.annotate('', xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle='->', color='#2E4057',
                                   lw=lw, mutation_scale=16), zorder=2)

    def arrow_h(ax, x1, x2, y, lw=2.0):
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color='#2E4057',
                                   lw=lw, mutation_scale=16), zorder=2)

    # 输入层
    block(ax, 2.0, 8.3, 2.2, 0.6, '输入图像\n416×416×3', fc='#BDD7EE', tc='#1F3864')
    arrow_v(ax, 2.0, 8.0, 7.5)

    # ========== Backbone 主干网络 ==========
    ax.text(0.5, 7.2, 'Backbone', fontsize=11, fontweight='bold',
            color='#C55A11', rotation=90, va='center')
    ax.add_patch(plt.Rectangle((0.1, 4.8), 0.25, 2.8, fc='#FFF2CC',
                                ec='#C55A11', lw=1.5, zorder=1))

    y = 7.2
    layers_backbone = [
        ('Conv\n208×208×32', '#ED7D31', 0.55),
        ('Conv\n104×104×64', '#ED7D31', 0.55),
        ('C2f (3)\n104×104×64', '#F4B183', 0.6),
        ('Conv\n52×52×128', '#ED7D31', 0.55),
        ('C2f (6)\n52×52×128', '#F4B183', 0.6),
        ('Conv\n26×26×256', '#ED7D31', 0.55),
        ('C2f (6)\n26×26×256', '#F4B183', 0.6),
        ('Conv\n13×13×512', '#ED7D31', 0.55),
        ('C2f (3)\n13×13×512', '#F4B183', 0.6),
        ('SPPF\n13×13×512', '#C65911', 0.6),
    ]

    for i, (text, color, h) in enumerate(layers_backbone):
        block(ax, 2.0, y, 1.8, h, text, fc=color, fs=7.5)
        if i < len(layers_backbone) - 1:
            arrow_v(ax, 2.0, y - h/2 - 0.05, y - h/2 - 0.35)
            y -= (h + 0.4)
        else:
            y -= h/2

    # ========== Neck 颈部网络 ==========
    ax.text(6.5, 7.2, 'Neck (PAN-FPN)', fontsize=11, fontweight='bold',
            color='#375623', rotation=90, va='center')
    ax.add_patch(plt.Rectangle((6.1, 3.5), 0.25, 4.0, fc='#E2EFDA',
                                ec='#70AD47', lw=1.5, zorder=1))

    # P5 -> Upsample
    arrow_h(ax, 2.9, 4.5, 4.5, lw=1.8)
    block(ax, 5.3, 4.5, 1.5, 0.5, 'Upsample\n×2', fc='#A9D08E', tc='#1F3864', fs=7.5)
    arrow_h(ax, 6.05, 7.0, 4.5, lw=1.8)

    # Concat + C2f
    block(ax, 7.8, 4.5, 1.4, 0.5, 'Concat', fc='#70AD47', fs=7.5)
    arrow_h(ax, 8.5, 9.2, 4.5, lw=1.8)
    block(ax, 10.0, 4.5, 1.5, 0.5, 'C2f (3)\n26×26×256', fc='#A9D08E', tc='#1F3864', fs=7.5)

    # P4 -> Upsample
    arrow_v(ax, 10.0, 4.2, 3.7, lw=1.8)
    block(ax, 10.0, 3.4, 1.5, 0.5, 'Upsample\n×2', fc='#A9D08E', tc='#1F3864', fs=7.5)
    arrow_v(ax, 10.0, 3.1, 2.6, lw=1.8)

    # Concat + C2f (P3)
    block(ax, 10.0, 2.3, 1.4, 0.5, 'Concat', fc='#70AD47', fs=7.5)
    arrow_v(ax, 10.0, 2.0, 1.5, lw=1.8)
    block(ax, 10.0, 1.2, 1.5, 0.5, 'C2f (3)\n52×52×128', fc='#A9D08E', tc='#1F3864', fs=7.5)

    # 下采样路径
    arrow_h(ax, 10.75, 11.3, 1.2, lw=1.8)
    block(ax, 11.8, 1.2, 1.3, 0.5, 'Conv\n26×26×256', fc='#ED7D31', fs=7.5)
    arrow_h(ax, 12.45, 12.9, 1.2, lw=1.8)

    # Concat + C2f (P4)
    block(ax, 12.9, 2.0, 0.4, 1.1, 'C', fc='#70AD47', fs=7)
    arrow_v(ax, 12.9, 2.55, 3.0, lw=1.8)
    block(ax, 12.9, 3.4, 1.5, 0.5, 'C2f (3)\n26×26×256', fc='#A9D08E', tc='#1F3864', fs=7.5)

    # 继续下采样
    arrow_v(ax, 12.9, 3.65, 4.1, lw=1.8)
    block(ax, 12.9, 4.4, 1.3, 0.5, 'Conv\n13×13×512', fc='#ED7D31', fs=7.5)
    arrow_v(ax, 12.9, 4.65, 5.1, lw=1.8)

    # Concat + C2f (P5)
    block(ax, 12.9, 5.8, 0.4, 1.1, 'C', fc='#70AD47', fs=7)
    arrow_v(ax, 12.9, 6.35, 6.8, lw=1.8)
    block(ax, 12.9, 7.2, 1.5, 0.5, 'C2f (3)\n13×13×512', fc='#A9D08E', tc='#1F3864', fs=7.5)

    # ========== Head 检测头 ==========
    ax.text(10.0, 0.3, 'Detection Head', fontsize=11, fontweight='bold',
            color='#4472C4', ha='center')

    # 三个检测输出
    arrow_v(ax, 10.0, 0.9, 0.6, lw=1.8)
    block(ax, 10.0, 0.3, 1.8, 0.4, 'P3: 52×52×3', fc='#5B9BD5', fs=7.5)

    arrow_v(ax, 12.9, 3.1, 2.8, lw=1.8)
    block(ax, 12.9, 2.5, 1.8, 0.4, 'P4: 26×26×3', fc='#5B9BD5', fs=7.5)

    arrow_v(ax, 12.9, 6.9, 6.6, lw=1.8)
    block(ax, 12.9, 6.3, 1.8, 0.4, 'P5: 13×13×3', fc='#5B9BD5', fs=7.5)

    # 图例
    legend_x = 0.8
    legend_y = 2.5
    ax.text(legend_x, legend_y + 0.3, '图例', fontsize=9, fontweight='bold', color='#1F3864')
    block(ax, legend_x + 0.5, legend_y - 0.2, 0.6, 0.3, 'Conv', fc='#ED7D31', fs=7)
    ax.text(legend_x + 1.0, legend_y - 0.2, '卷积层', fontsize=7, va='center', color='#1F3864')
    block(ax, legend_x + 0.5, legend_y - 0.6, 0.6, 0.3, 'C2f', fc='#F4B183', fs=7)
    ax.text(legend_x + 1.0, legend_y - 0.6, 'C2f模块', fontsize=7, va='center', color='#1F3864')
    block(ax, legend_x + 0.5, legend_y - 1.0, 0.6, 0.3, 'SPPF', fc='#C65911', fs=7)
    ax.text(legend_x + 1.0, legend_y - 1.0, '空间金字塔', fontsize=7, va='center', color='#1F3864')

    ax.set_title('图7  YOLOv8n网络结构图', fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图7_YOLOv8n网络结构图.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图7 完成")


# ============================================================
# 图8: ECA-Net模块结构示意图（概览）
# ============================================================
def fig8_eca_overview():
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis('off')

    def box(ax, x, y, w, h, text, fc='#4472C4', tc='white', fs=9):
        b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                            boxstyle="round,pad=0.08,rounding_size=0.15",
                            linewidth=1.3, edgecolor='#1F3864', facecolor=fc, zorder=3)
        ax.add_patch(b)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fs, color=tc, fontweight='bold', zorder=4)

    def arrow(ax, x1, x2, y=2.0):
        ax.annotate('', xy=(x2 - 0.05, y), xytext=(x1 + 0.05, y),
                    arrowprops=dict(arrowstyle='->', color='#2E4057',
                                   lw=2.0, mutation_scale=18), zorder=2)

    # 输入特征图
    ax.text(0.7, 2.0, 'Z\nC×H×W', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#1F3864',
            bbox=dict(boxstyle='round,pad=0.4', fc='#BDD7EE', ec='#2E75B6', lw=1.5))

    # 全局平均池化
    arrow(ax, 1.3, 2.0)
    box(ax, 2.9, 2.0, 2.4, 0.85, '全局平均池化\nGlobal Avg Pool\n→ C×1×1', fc='#ED7D31')

    # 一维卷积
    arrow(ax, 4.1, 4.7)
    box(ax, 5.6, 2.0, 2.2, 0.85, '自适应1D卷积\nConv1D (k)\n跨通道交互', fc='#4472C4')

    # Sigmoid
    arrow(ax, 6.7, 7.3)
    box(ax, 8.1, 2.0, 1.8, 0.85, 'Sigmoid\n激活函数\n→ ω ∈(0,1)', fc='#5B9BD5')

    # 乘法操作
    arrow(ax, 9.0, 9.6)
    circ = plt.Circle((9.9, 2.0), 0.3, color='#70AD47', zorder=3, ec='#375623', lw=1.5)
    ax.add_patch(circ)
    ax.text(9.9, 2.0, '⊗', ha='center', va='center', fontsize=16,
            color='white', fontweight='bold', zorder=4)

    # 与原始特征图相乘
    ax.annotate('', xy=(9.9, 1.35), xytext=(9.9, 0.5),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5))
    ax.text(9.9, 0.3, '原始特征图 Z', ha='center', fontsize=8.5, color='#444444')

    # 输出
    arrow(ax, 10.2, 11.0)
    ax.text(11.8, 2.0, 'Z̃\n(重标定特征)\nC×H×W', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#1F3864',
            bbox=dict(boxstyle='round,pad=0.4', fc='#E2EFDA', ec='#70AD47', lw=1.5))

    # 公式标注
    ax.text(6.5, 3.6, 'ω = σ(Conv1D_k(GAP(Z)))     Z̃ = ω ⊗ Z',
            ha='center', fontsize=10, color='#1F3864',
            bbox=dict(boxstyle='round,pad=0.4', fc='#F9F9F9', ec='#AAAAAA'))

    ax.set_title('图8  ECA-Net模块结构示意图（概览）', fontsize=13, fontweight='bold', pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图8_ECA-Net模块结构示意图.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图8 完成")


# ============================================================
# 图9: YOLOv8n颈部特征融合结构示意图
# ============================================================
def fig9_neck_structure():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis('off')

    def block(ax, x, y, w, h, text, fc='#4472C4', tc='white', fs=8.5, r=0.2):
        b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                            boxstyle=f"round,pad=0.05,rounding_size={r}",
                            linewidth=1.2, edgecolor='#1F3864', facecolor=fc, zorder=3)
        ax.add_patch(b)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fs, color=tc, fontweight='bold', zorder=4)

    def arr(ax, x1, y1, x2, y2, color='#555555'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=1.5, mutation_scale=14), zorder=2)

    # Backbone 输出特征图（三个尺度）
    block(ax, 1.2, 5.8, 1.6, 0.65, 'P3\n52×52×128\n(浅层特征)', fc='#2E75B6')
    block(ax, 1.2, 3.5, 1.6, 0.65, 'P4\n26×26×256\n(中层特征)', fc='#2E75B6')
    block(ax, 1.2, 1.2, 1.6, 0.65, 'P5\n13×13×512\n(深层语义)', fc='#1F3864')

    # 区域标签
    ax.text(0.4, 6.6, 'Backbone', fontsize=9, color='#2E75B6',
            fontweight='bold', rotation=90, va='center')
    ax.text(4.5, 6.6, 'Neck (PAN-FPN)', fontsize=9, color='#C55A11',
            fontweight='bold', va='center')
    ax.text(10.5, 6.6, 'Head', fontsize=9, color='#375623',
            fontweight='bold', va='center')

    # FPN: 自顶向下路径
    block(ax, 4.5, 1.2, 1.4, 0.55, 'SPPF', fc='#ED7D31', fs=8)
    arr(ax, 2.0, 1.2, 3.8, 1.2)

    block(ax, 4.5, 3.5, 1.6, 0.55, 'Upsample\n+Concat', fc='#FFC000', tc='#1F3864', fs=8)
    arr(ax, 4.5, 1.47, 4.5, 3.22)   # P5→upsample
    arr(ax, 2.0, 3.5, 3.7, 3.5)     # P4→concat

    block(ax, 4.5, 5.8, 1.6, 0.55, 'Upsample\n+Concat', fc='#FFC000', tc='#1F3864', fs=8)
    arr(ax, 4.5, 3.77, 4.5, 5.52)
    arr(ax, 2.0, 5.8, 3.7, 5.8)

    # C2f 模块
    block(ax, 6.5, 5.8, 1.3, 0.55, 'C2f', fc='#4472C4', fs=9)
    arr(ax, 5.3, 5.8, 5.85, 5.8)

    block(ax, 6.5, 3.5, 1.3, 0.55, 'C2f', fc='#4472C4', fs=9)
    arr(ax, 5.3, 3.5, 5.85, 3.5)

    # PAN: 自底向上路径
    block(ax, 8.0, 3.5, 1.6, 0.55, 'Downsample\n+Concat', fc='#FFC000', tc='#1F3864', fs=8)
    arr(ax, 7.15, 5.8, 8.0, 3.77)   # 从小目标到中目标
    arr(ax, 7.15, 3.5, 7.2, 3.5)

    block(ax, 8.0, 1.2, 1.6, 0.55, 'Downsample\n+Concat', fc='#FFC000', tc='#1F3864', fs=8)
    arr(ax, 8.0, 3.22, 8.0, 1.47)
    arr(ax, 5.3, 1.2, 7.2, 1.2)

    # C2f 模块（输出前）
    block(ax, 9.8, 5.8, 1.3, 0.55, 'C2f', fc='#4472C4', fs=9)
    arr(ax, 7.15, 5.8, 9.15, 5.8)   # 跨越

    block(ax, 9.8, 3.5, 1.3, 0.55, 'C2f', fc='#4472C4', fs=9)
    arr(ax, 8.8, 3.5, 9.15, 3.5)

    block(ax, 9.8, 1.2, 1.3, 0.55, 'C2f', fc='#4472C4', fs=9)
    arr(ax, 8.8, 1.2, 9.15, 1.2)

    # 检测头输出
    block(ax, 11.5, 5.8, 1.5, 0.55, 'Detect\n(小目标)\n52×52', fc='#70AD47', fs=8)
    arr(ax, 10.45, 5.8, 10.75, 5.8)

    block(ax, 11.5, 3.5, 1.5, 0.55, 'Detect\n(中目标)\n26×26', fc='#70AD47', fs=8)
    arr(ax, 10.45, 3.5, 10.75, 3.5)

    block(ax, 11.5, 1.2, 1.5, 0.55, 'Detect\n(大目标)\n13×13', fc='#70AD47', fs=8)
    arr(ax, 10.45, 1.2, 10.75, 1.2)

    # 标注问题点
    ax.text(4.5, 6.5, '⚠ 简单Concat忽略通道重要性',
            ha='center', fontsize=7.5, color='#C00000', style='italic')

    ax.set_title('图9  YOLOv8n颈部特征融合结构示意图', fontsize=13, fontweight='bold', pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图9_YOLOv8n颈部特征融合结构示意图.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图9 完成")


# ============================================================
# 图10: ECA模块详细结构图
# ============================================================
def fig10_eca_detail():
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    def box(ax, x, y, w, h, text, fc='#4472C4', tc='white', fs=9):
        b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                            boxstyle="round,pad=0.06,rounding_size=0.15",
                            linewidth=1.3, edgecolor='#1F3864', facecolor=fc, zorder=3)
        ax.add_patch(b)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fs, color=tc, fontweight='bold', zorder=4)

    def arr(ax, x1, y1, x2, y2, label='', color='#333333'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=1.8, mutation_scale=16), zorder=2)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.1, my + 0.15, label, fontsize=7.5,
                    color='#555555', ha='center', style='italic')

    y_main = 2.8

    # 输入
    ax.text(0.75, y_main, 'Z\nC×H×W', ha='center', va='center', fontsize=9.5,
            fontweight='bold', color='#1F3864',
            bbox=dict(boxstyle='round,pad=0.4', fc='#BDD7EE', ec='#2E75B6', lw=1.8))

    arr(ax, 1.3, y_main, 2.0, y_main, 'Z')

    # 步骤1: 全局平均池化
    box(ax, 3.1, y_main, 2.0, 1.1,
        '①全局平均池化\nAdaptive AvgPool2D\nC×H×W → C×1×1', fc='#ED7D31')
    arr(ax, 2.0, y_main, 2.1, y_main)

    # 步骤2: reshape
    arr(ax, 4.1, y_main, 4.8, y_main, 'C×1×1')
    box(ax, 5.5, y_main, 1.2, 0.75, 'Reshape\n→ 1×C', fc='#7B7B7B', fs=8.5)

    # 步骤3: 1D卷积
    arr(ax, 6.1, y_main, 6.8, y_main, '1×C')
    box(ax, 7.9, y_main, 2.0, 1.1,
        '②自适应1D卷积\nConv1d(1,1,k)\n捕获局部跨通道依赖', fc='#4472C4')

    # k值公式
    ax.text(7.9, 4.4, 'k = |log₂(C)/2 + 0.5|_odd',
            ha='center', fontsize=8.5, color='#C00000',
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFF2E2', ec='#ED7D31'))
    ax.annotate('', xy=(7.9, 3.35), xytext=(7.9, 4.15),
                arrowprops=dict(arrowstyle='->', color='#ED7D31', lw=1.2))

    # 步骤4: Sigmoid
    arr(ax, 8.9, y_main, 9.6, y_main, '1×C')
    box(ax, 10.4, y_main, 1.5, 0.9,
        '③Sigmoid\n激活\nω∈(0,1)', fc='#5B9BD5')

    # reshape回C×1×1
    arr(ax, 11.15, y_main, 11.7, y_main, 'ω')
    box(ax, 12.2, y_main, 1.0, 0.75, 'Reshape\nC×1×1', fc='#7B7B7B', fs=8.5)

    # 乘法
    arr(ax, 12.7, y_main, 13.15, y_main)
    circ = plt.Circle((13.4, y_main), 0.28, color='#70AD47', zorder=3,
                       ec='#375623', lw=1.5)
    ax.add_patch(circ)
    ax.text(13.4, y_main, '⊗', ha='center', va='center',
            fontsize=15, color='white', fontweight='bold', zorder=4)

    # 原始输入Z连接到乘法节点
    ax.plot([0.75, 0.75, 13.4, 13.4], [y_main, 1.1, 1.1, y_main - 0.28],
            '-', color='#AAAAAA', lw=1.5, zorder=2)
    ax.text(7.0, 0.7, '原始特征图 Z (跳跃连接)', ha='center',
            fontsize=8, color='#777777', style='italic')

    # 输出
    ax.text(13.4, y_main + 1.0, 'Z̃ = ω⊗Z\n输出特征图', ha='center',
            fontsize=8.5, color='#375623', fontweight='bold')

    ax.set_title('图10  ECA-Net模块详细结构示意图', fontsize=13, fontweight='bold', pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图10_ECA模块详细结构图.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图10 完成")


# ============================================================
# 图11: ECA模块核心代码展示
# ============================================================
def fig11_eca_code():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # 代码背景
    code_bg = FancyBboxPatch((0.3, 0.3), 11.4, 6.4,
                              boxstyle="round,pad=0.1,rounding_size=0.2",
                              linewidth=1.5, edgecolor='#AAAAAA',
                              facecolor='#1E1E2E', zorder=1)
    ax.add_patch(code_bg)

    # 标题栏
    title_bg = FancyBboxPatch((0.3, 6.3), 11.4, 0.4,
                               boxstyle="round,pad=0.0",
                               linewidth=0, facecolor='#2D2D44', zorder=2)
    ax.add_patch(title_bg)
    ax.text(0.6, 6.5, '●', fontsize=10, color='#FF5F57', zorder=3)
    ax.text(0.95, 6.5, '●', fontsize=10, color='#FEBC2E', zorder=3)
    ax.text(1.3, 6.5, '●', fontsize=10, color='#28C840', zorder=3)
    ax.text(6.0, 6.5, 'eca_module.py', ha='center', fontsize=9,
            color='#AAAAAA', zorder=3)

    code_lines = [
        ('import torch', '#CC99CD', 10),
        ('import torch.nn as nn', '#CC99CD', 10),
        ('import math', '#CC99CD', 10),
        ('', '', 9),
        ('class ECA(nn.Module):', '#569CD6', 10.5),
        ('    """ECA-Net: Efficient Channel Attention Module', '#6A9955', 9.5),
        ('    Wang et al., CVPR 2020"""', '#6A9955', 9.5),
        ('    def __init__(self, channels, gamma=2, b=1):', '#DCDCAA', 10),
        ('        super(ECA, self).__init__()', '#D4D4D4', 9.5),
        ('        # 自适应计算卷积核大小 k', '#6A9955', 9),
        ('        t = int(abs(math.log(channels, 2) + b) / gamma)', '#D4D4D4', 9.5),
        ('        k = t if t % 2 else t + 1  # 保证 k 为奇数', '#D4D4D4', 9.5),
        ('        self.avg_pool = nn.AdaptiveAvgPool2d(1)', '#D4D4D4', 9.5),
        ('        self.conv = nn.Conv1d(1, 1, kernel_size=k,', '#D4D4D4', 9.5),
        ('                             padding=(k-1)//2, bias=False)', '#D4D4D4', 9.5),
        ('        self.sigmoid = nn.Sigmoid()', '#D4D4D4', 9.5),
        ('', '', 9),
        ('    def forward(self, x):', '#DCDCAA', 10),
        ('        y = self.avg_pool(x)                   # [B,C,1,1]', '#D4D4D4', 9.5),
        ('        y = y.squeeze(-1).transpose(-1, -2)    # [B,1,C]', '#D4D4D4', 9.5),
        ('        y = self.conv(y)                       # [B,1,C]', '#D4D4D4', 9.5),
        ('        y = self.sigmoid(y)', '#D4D4D4', 9.5),
        ('        y = y.transpose(-1,-2).unsqueeze(-1)   # [B,C,1,1]', '#D4D4D4', 9.5),
        ('        return x * y.expand_as(x)              # 通道重标定', '#D4D4D4', 9.5),
    ]

    y_pos = 6.05
    for line, color, fs in code_lines:
        if line:  # 跳过空行的文字绘制，颜色可能为空
            ax.text(0.55, y_pos, line, fontsize=fs, color=color,
                    fontfamily='monospace', va='center', zorder=3)
        y_pos -= 0.245

    ax.set_title('图11  ECA-Net模块核心代码', fontsize=13, fontweight='bold',
                 color='#1F3864', pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图11_ECA模块核心代码.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图11 完成")


# ============================================================
# 图12: YOLOv8n+ECA改进后网络结构图
# ============================================================
def fig12_improved_network():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    def block(ax, x, y, w, h, text, fc='#4472C4', tc='white', fs=8.5, r=0.18):
        b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                            boxstyle=f"round,pad=0.05,rounding_size={r}",
                            linewidth=1.2, edgecolor='#1F3864', facecolor=fc, zorder=3)
        ax.add_patch(b)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fs, color=tc, fontweight='bold', zorder=4)

    def eca_block(ax, x, y, w=1.3, h=0.55):
        """红色高亮的ECA模块"""
        b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                            boxstyle="round,pad=0.05,rounding_size=0.15",
                            linewidth=2.0, edgecolor='#C00000',
                            facecolor='#FFD966', zorder=3)
        ax.add_patch(b)
        ax.text(x, y, '★ ECA\n注意力', ha='center', va='center',
                fontsize=8, color='#7B3F00', fontweight='bold', zorder=4)

    def arr(ax, x1, y1, x2, y2, color='#555555'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=1.4, mutation_scale=13), zorder=2)

    # ---- Backbone ----
    ax.text(1.3, 7.6, 'Backbone', fontsize=10, fontweight='bold',
            color='#1F3864', ha='center')
    backbone_blocks = [
        (1.3, 6.9, 'Conv\n(640→320)'),
        (1.3, 6.1, 'Conv\n(320→160)'),
        (1.3, 5.2, 'C2f × 3\n(160×160)'),
        (1.3, 4.2, 'Conv\n(160→80)'),
        (1.3, 3.3, 'C2f × 6\n(80×80)'),
        (1.3, 2.4, 'Conv\n(80→40)'),
        (1.3, 1.5, 'C2f × 6\n(40×40)'),
        (1.3, 0.75, 'SPPF\n(13×13)'),
    ]
    for (x, y, t) in backbone_blocks:
        block(ax, x, y, 1.7, 0.58, t, fc='#2E75B6', fs=8)
    for i in range(len(backbone_blocks) - 1):
        arr(ax, backbone_blocks[i][0], backbone_blocks[i][1] - 0.29,
            backbone_blocks[i+1][0], backbone_blocks[i+1][1] + 0.29)

    # ---- Neck ----
    ax.text(6.5, 7.6, 'Neck (PAN-FPN + ECA改进)',
            fontsize=10, fontweight='bold', color='#C55A11', ha='center')

    # FPN 自顶向下
    block(ax, 5.0, 0.75, 1.6, 0.55, 'C2f\n13×13', fc='#4472C4', fs=8)
    arr(ax, 2.1, 0.75, 4.2, 0.75)

    # ECA → Upsample → Concat (第一个改进点)
    eca_block(ax, 5.0, 2.0)
    arr(ax, 5.0, 1.02, 5.0, 1.72)
    block(ax, 5.0, 2.75, 1.6, 0.55, 'Upsample\n→26×26', fc='#FFC000', tc='#1F3864', fs=8)
    arr(ax, 5.0, 2.27, 5.0, 2.47)
    # 来自Backbone C2f(80×80)的连接
    arr(ax, 2.1, 3.3, 4.2, 2.75)
    block(ax, 5.0, 3.5, 1.6, 0.55, 'Concat\n26×26', fc='#FFC000', tc='#1F3864', fs=8)
    arr(ax, 5.0, 3.02, 5.0, 3.22)
    block(ax, 5.0, 4.3, 1.6, 0.58, 'C2f\n26×26', fc='#4472C4', fs=8)
    arr(ax, 5.0, 3.77, 5.0, 4.01)

    # ECA → Upsample → Concat (第二个改进点)
    eca_block(ax, 5.0, 5.15)
    arr(ax, 5.0, 4.59, 5.0, 4.87)
    block(ax, 5.0, 5.85, 1.6, 0.55, 'Upsample\n→52×52', fc='#FFC000', tc='#1F3864', fs=8)
    arr(ax, 5.0, 5.42, 5.0, 5.57)
    arr(ax, 2.1, 5.2, 4.2, 5.85)  # 来自C2f(160×160)
    block(ax, 5.0, 6.55, 1.6, 0.55, 'Concat\n52×52', fc='#FFC000', tc='#1F3864', fs=8)
    arr(ax, 5.0, 6.12, 5.0, 6.27)
    block(ax, 5.0, 7.3, 1.6, 0.58, 'C2f\n52×52', fc='#4472C4', fs=8)
    arr(ax, 5.0, 6.82, 5.0, 7.01)

    # PAN 自底向上
    block(ax, 8.0, 7.3, 1.6, 0.58, 'C2f\n52×52', fc='#4472C4', fs=8)
    arr(ax, 5.8, 7.3, 7.2, 7.3)

    block(ax, 8.0, 6.1, 1.6, 0.55, 'Conv\n+Concat\n26×26', fc='#FFC000', tc='#1F3864', fs=8)
    arr(ax, 8.0, 7.01, 8.0, 6.37)
    arr(ax, 5.8, 4.3, 7.2, 6.1)  # 来自C2f(26×26)

    block(ax, 8.0, 4.9, 1.6, 0.58, 'C2f\n26×26', fc='#4472C4', fs=8)
    arr(ax, 8.0, 5.82, 8.0, 5.19)

    block(ax, 8.0, 3.7, 1.6, 0.55, 'Conv\n+Concat\n13×13', fc='#FFC000', tc='#1F3864', fs=8)
    arr(ax, 8.0, 4.61, 8.0, 3.97)
    arr(ax, 5.8, 0.75, 7.2, 3.7)  # 来自SPPF

    block(ax, 8.0, 2.5, 1.6, 0.58, 'C2f\n13×13', fc='#4472C4', fs=8)
    arr(ax, 8.0, 3.42, 8.0, 2.79)

    # ---- Head ----
    ax.text(11.8, 7.6, 'Head', fontsize=10, fontweight='bold',
            color='#375623', ha='center')
    block(ax, 11.5, 7.3, 2.0, 0.58, 'Detect (小目标)\n52×52 → 3类', fc='#70AD47', fs=8)
    block(ax, 11.5, 4.9, 2.0, 0.58, 'Detect (中目标)\n26×26 → 3类', fc='#70AD47', fs=8)
    block(ax, 11.5, 2.5, 2.0, 0.58, 'Detect (大目标)\n13×13 → 3类', fc='#70AD47', fs=8)
    arr(ax, 8.8, 7.3, 10.5, 7.3)
    arr(ax, 8.8, 4.9, 10.5, 4.9)
    arr(ax, 8.8, 2.5, 10.5, 2.5)

    # 图例
    legend_x, legend_y = 9.5, 1.4
    eca_leg = FancyBboxPatch((legend_x, legend_y - 0.22), 0.7, 0.42,
                              boxstyle="round,pad=0.02",
                              facecolor='#FFD966', edgecolor='#C00000', lw=2.0, zorder=3)
    ax.add_patch(eca_leg)
    ax.text(legend_x + 0.35, legend_y, '★ ECA', ha='center', va='center',
            fontsize=7.5, color='#7B3F00', fontweight='bold', zorder=4)
    ax.text(legend_x + 0.85, legend_y, '= 新增ECA注意力模块（改进点）',
            fontsize=8, color='#C00000', va='center')

    ax.set_title('图12  YOLOv8n+ECA改进后网络结构图',
                 fontsize=13, fontweight='bold', pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图12_YOLOv8n+ECA网络结构图.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图12 完成")


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print("开始生成毕业设计科研图...")
    print(f"输出目录: {OUTPUT_DIR}\n")

    fig1_detection_tree()
    fig2_traditional_flowchart()
    fig3_cnn_structure()
    fig4_convolution()
    fig5_fully_connected()
    fig6_yolov8_comparison()
    fig7_yolov8n_structure()
    fig8_eca_overview()
    fig9_neck_structure()
    fig10_eca_detail()
    fig11_eca_code()
    fig12_improved_network()

    print("\n全部12张科研图生成完成！")
    print("请前往查看: " + OUTPUT_DIR)
