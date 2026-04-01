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
    "cheekPuff",
]

def params_dict_to_vector(params_dict):
    return [float(params_dict.get(k, 0.0)) for k in PARAM_ORDER]

def vector_to_params_dict(vector):
    return {k: float(vector[i]) for i, k in enumerate(PARAM_ORDER)}

def clamp_params_dict(params_dict):
    clamped = {}
    for k, v in params_dict.items():
        if "eyeLook" in k:
            clamped[k] = max(-1.0, min(1.0, float(v)))
        else:
            clamped[k] = max(0.0, min(1.0, float(v)))
    return clamped