# 参数顺序固定
PARAM_ORDER = [
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "browDownLeft",
    "browDownRight",
    "eyeWideLeft",
    "eyeWideRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeLookUp",
    "eyeLookDown",
    "eyeLookLeft",
    "eyeLookRight",
    "jawOpen",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "cheekPuff"
]

def rule_mapping(emotion, intensity):
    """
    返回一个长度为 len(PARAM_ORDER) 的列表，每个元素对应 PARAM_ORDER 中的参数值。
    参数值范围：眼球方向为 -1~1，其他 blendshape 为 0~1。
    """
    # 初始化参数字典，默认所有值为 0.0
    params_dict = {key: 0.0 for key in PARAM_ORDER}
    
    # 确保强度在 [0,1] 范围内
    intensity = max(0.0, min(1.0, intensity))

    if emotion == "happy":
        params_dict["mouthSmileLeft"] = 0.7 * intensity
        params_dict["mouthSmileRight"] = 0.7 * intensity
        params_dict["eyeSquintLeft"] = 0.4 * intensity
        params_dict["eyeSquintRight"] = 0.4 * intensity
        params_dict["browInnerUp"] = 0.2 * intensity

    elif emotion == "sad":
        params_dict["mouthFrownLeft"] = 0.7 * intensity
        params_dict["mouthFrownRight"] = 0.7 * intensity
        params_dict["browInnerUp"] = 0.6 * intensity
        params_dict["eyeSquintLeft"] = 0.3 * intensity
        params_dict["eyeSquintRight"] = 0.3 * intensity

    elif emotion == "angry":
        params_dict["browDownLeft"] = 0.8 * intensity
        params_dict["browDownRight"] = 0.8 * intensity
        params_dict["mouthPressLeft"] = 0.6 * intensity
        params_dict["mouthPressRight"] = 0.6 * intensity
        params_dict["cheekPuff"] = 0.3 * intensity

    elif emotion == "surprise":
        params_dict["browOuterUpLeft"] = 0.8 * intensity
        params_dict["browOuterUpRight"] = 0.8 * intensity
        params_dict["eyeWideLeft"] = 0.9 * intensity
        params_dict["eyeWideRight"] = 0.9 * intensity
        params_dict["jawOpen"] = 0.6 * intensity

    elif emotion == "thoughtful":
        params_dict["browInnerUp"] = 0.4 * intensity
        params_dict["eyeLookDown"] = 0.3 * intensity
        params_dict["mouthPressLeft"] = 0.4 * intensity
        params_dict["mouthPressRight"] = 0.4 * intensity

    elif emotion == "disgust":
        params_dict["browDownLeft"] = 0.5 * intensity
        params_dict["browDownRight"] = 0.5 * intensity
        params_dict["eyeSquintLeft"] = 0.6 * intensity
        params_dict["eyeSquintRight"] = 0.6 * intensity
        params_dict["mouthFrownLeft"] = 0.5 * intensity
        params_dict["mouthFrownRight"] = 0.5 * intensity
        params_dict["cheekPuff"] = 0.2 * intensity

    elif emotion == "bored":
        params_dict["eyeLookDown"] = 0.4 * intensity
        params_dict["eyeLookLeft"] = 0.3 * intensity
        params_dict["mouthFrownLeft"] = 0.2 * intensity
        params_dict["mouthFrownRight"] = 0.2 * intensity
        params_dict["browDownLeft"] = 0.3 * intensity
        params_dict["browDownRight"] = 0.3 * intensity

    elif emotion == "concern":
        params_dict["browInnerUp"] = 0.6 * intensity
        params_dict["eyeWideLeft"] = 0.4 * intensity
        params_dict["eyeWideRight"] = 0.4 * intensity
        params_dict["mouthPressLeft"] = 0.3 * intensity
        params_dict["mouthPressRight"] = 0.3 * intensity

    elif emotion == "tired":
        params_dict["eyeLookDown"] = 0.5 * intensity
        params_dict["eyeSquintLeft"] = 0.4 * intensity
        params_dict["eyeSquintRight"] = 0.4 * intensity
        params_dict["mouthFrownLeft"] = 0.3 * intensity
        params_dict["mouthFrownRight"] = 0.3 * intensity
        params_dict["browDownLeft"] = 0.4 * intensity
        params_dict["browDownRight"] = 0.4 * intensity

    elif emotion == "calm":
        params_dict["mouthSmileLeft"] = 0.1 * intensity
        params_dict["mouthSmileRight"] = 0.1 * intensity

    # 对眼球方向进行范围限制（-1~1）
    for key in ["eyeLookUp", "eyeLookDown", "eyeLookLeft", "eyeLookRight"]:
        params_dict[key] = max(-1.0, min(1.0, params_dict[key]))

    # 转换为固定顺序的列表
    vector = [params_dict[key] for key in PARAM_ORDER]
    return vector