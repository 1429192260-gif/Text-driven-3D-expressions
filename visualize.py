import matplotlib.pyplot as plt

def draw_face(params, title=""):

    fig, ax = plt.subplots()

    # 头
    circle = plt.Circle((0, 0), 1, fill=False)
    ax.add_patch(circle)

    # 眼睛
    eye_y = 0.3
    eye_offset = 0.4

    eye_open = params.get("eyeWideLeft", 0.2)
    eye_size = 0.1 + 0.1 * eye_open

    ax.add_patch(plt.Circle((-eye_offset, eye_y), eye_size))
    ax.add_patch(plt.Circle((eye_offset, eye_y), eye_size))

    # 眉毛
    brow = params.get("browInnerUp", 0.0)
    ax.plot([-0.6, -0.2], [0.5 + brow, 0.5 + brow])
    ax.plot([0.2, 0.6], [0.5 + brow, 0.5 + brow])

    # 嘴巴（关键）
    smile = params.get("mouthSmileLeft", 0) - params.get("mouthFrownLeft", 0)

    x = [-0.5, 0, 0.5]
    y = [-0.3 + smile, -0.3 - smile, -0.3 + smile]

    ax.plot(x, y)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title(title)
    ax.axis('off')

    plt.show()