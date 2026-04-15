# Case Analysis: Text-Only Front-Ends

- setting: manual-only source-holdout test set
- cases: 16

## Mean Case Param MAE

| Front-End | Mean Case Param MAE |
| --- | ---: |
| learned | 0.067369 |
| hf_raw | 0.074353 |
| hf_rules_v4 | 0.019119 |

## Case 1: happy

- text: 今天终于有件事按我想的来了，心情不错。
- true_emotion: happy
- true_intensity: 0.61

| Variant | Pred Emotion | Pred Intensity | Confidence | Param MAE | Raw Label |
| --- | --- | ---: | ---: | ---: | --- |
| learned | happy | 0.68 | 0.9998 | 0.005106 | happy |
| hf_raw | happy | 0.64 | 0.9894 | 0.001995 | happy |
| hf_rules_v4 | happy | 0.64 | 0.9894 | 0.001995 | happy |

Top parameter differences (`hf_rules_v4`):
- mouthSmileRight: pred=0.4174, target=0.4, abs_diff=0.0174
- eyeSquintLeft: pred=0.1702, target=0.16, abs_diff=0.0102
- eyeSquintRight: pred=0.1768, target=0.17, abs_diff=0.0068
- browInnerUp: pred=0.0844, target=0.09, abs_diff=0.0056
- mouthSmileLeft: pred=0.4061, target=0.41, abs_diff=0.0039

## Case 2: happy

- text: 哈哈，真的没白等，我现在特别开心。
- true_emotion: happy
- true_intensity: 0.82

| Variant | Pred Emotion | Pred Intensity | Confidence | Param MAE | Raw Label |
| --- | --- | ---: | ---: | ---: | --- |
| learned | happy | 0.76 | 0.9929 | 0.013424 | happy |
| hf_raw | happy | 0.72 | 0.9908 | 0.017390 | happy |
| hf_rules_v4 | happy | 0.77 | 0.9908 | 0.012431 | happy |

Top parameter differences (`hf_rules_v4`):
- mouthSmileRight: pred=0.4918, target=0.58, abs_diff=0.0882
- mouthSmileLeft: pred=0.5027, target=0.58, abs_diff=0.0773
- eyeSquintRight: pred=0.2154, target=0.26, abs_diff=0.0446
- eyeSquintLeft: pred=0.2234, target=0.26, abs_diff=0.0366
- browInnerUp: pred=0.0932, target=0.12, abs_diff=0.0268

## Case 3: sad

- text: 整个人都提不起精神。#3
- true_emotion: sad
- true_intensity: 0.45

| Variant | Pred Emotion | Pred Intensity | Confidence | Param MAE | Raw Label |
| --- | --- | ---: | ---: | ---: | --- |
| learned | sad | 0.54 | 0.7299 | 0.012311 | sad |
| hf_raw | angry | 0.70 | 0.4868 | 0.133993 | angry |
| hf_rules_v4 | sad | 0.58 | 0.4868 | 0.016792 | angry |

Top parameter differences (`hf_rules_v4`):
- mouthFrownRight: pred=0.3624, target=0.26, abs_diff=0.1024
- mouthFrownLeft: pred=0.371, target=0.28, abs_diff=0.091
- browInnerUp: pred=0.3244, target=0.24, abs_diff=0.0844
- eyeSquintRight: pred=0.1512, target=0.1, abs_diff=0.0512
- eyeSquintLeft: pred=0.1604, target=0.12, abs_diff=0.0404

## Case 4: sad

- text: 我现在一点都高兴不起来，越想越委屈。
- true_emotion: sad
- true_intensity: 0.86

| Variant | Pred Emotion | Pred Intensity | Confidence | Param MAE | Raw Label |
| --- | --- | ---: | ---: | ---: | --- |
| learned | surprise | 0.69 | 0.8825 | 0.204003 | surprise |
| hf_raw | sad | 0.50 | 0.9844 | 0.050005 | sad |
| hf_rules_v4 | sad | 0.68 | 0.9844 | 0.029626 | sad |

Top parameter differences (`hf_rules_v4`):
- mouthFrownRight: pred=0.4146, target=0.6, abs_diff=0.1854
- mouthFrownLeft: pred=0.4102, target=0.59, abs_diff=0.1798
- browInnerUp: pred=0.3578, target=0.48, abs_diff=0.1222
- eyeSquintRight: pred=0.1675, target=0.25, abs_diff=0.0825
- eyeSquintLeft: pred=0.1582, target=0.24, abs_diff=0.0818

## Case 5: angry

- text: 这事让我有点不舒服，心里直犯躁。
- true_emotion: angry
- true_intensity: 0.39

| Variant | Pred Emotion | Pred Intensity | Confidence | Param MAE | Raw Label |
| --- | --- | ---: | ---: | ---: | --- |
| learned | surprise | 0.69 | 0.9000 | 0.148081 | surprise |
| hf_raw | disgust | 0.57 | 0.9631 | 0.081959 | disgust |
| hf_rules_v4 | angry | 0.62 | 0.9631 | 0.038284 | disgust |

Top parameter differences (`hf_rules_v4`):
- browDownLeft: pred=0.4544, target=0.23, abs_diff=0.2244
- browDownRight: pred=0.4443, target=0.22, abs_diff=0.2243
- mouthPressLeft: pred=0.3337, target=0.16, abs_diff=0.1737
- mouthPressRight: pred=0.3307, target=0.17, abs_diff=0.1607
- cheekPuff: pred=0.1291, target=0.07, abs_diff=0.0591

## Case 6: angry

- text: 再这样下去我就要爆发了。#8
- true_emotion: angry
- true_intensity: 0.79

| Variant | Pred Emotion | Pred Intensity | Confidence | Param MAE | Raw Label |
| --- | --- | ---: | ---: | ---: | --- |
| learned | calm | 0.42 | 0.9758 | 0.104549 | calm |
| hf_raw | sad | 0.55 | 0.5832 | 0.156963 | sad |
| hf_rules_v4 | angry | 0.68 | 0.5832 | 0.015983 | sad |

Top parameter differences (`hf_rules_v4`):
- browDownLeft: pred=0.4826, target=0.57, abs_diff=0.0874
- mouthPressLeft: pred=0.3523, target=0.43, abs_diff=0.0777
- browDownRight: pred=0.5035, target=0.58, abs_diff=0.0765
- mouthPressRight: pred=0.3702, target=0.44, abs_diff=0.0698
- cheekPuff: pred=0.1497, target=0.19, abs_diff=0.0403

## Case 7: surprise

- text: 咦？这和我想的有点不一样。
- true_emotion: surprise
- true_intensity: 0.31

| Variant | Pred Emotion | Pred Intensity | Confidence | Param MAE | Raw Label |
| --- | --- | ---: | ---: | ---: | --- |
| learned | surprise | 0.71 | 0.5614 | 0.069590 | surprise |
| hf_raw | calm | 0.37 | 0.9463 | 0.048448 | calm |
| hf_rules_v4 | surprise | 0.64 | 0.9463 | 0.057645 | calm |

Top parameter differences (`hf_rules_v4`):
- eyeWideLeft: pred=0.5266, target=0.23, abs_diff=0.2966
- eyeWideRight: pred=0.5111, target=0.24, abs_diff=0.2711
- browOuterUpRight: pred=0.4319, target=0.17, abs_diff=0.2619
- browOuterUpLeft: pred=0.4336, target=0.18, abs_diff=0.2536
- jawOpen: pred=0.2851, target=0.1, abs_diff=0.1851

## Case 8: surprise

- text: 天啊，我一下子都不知道该说什么了。
- true_emotion: surprise
- true_intensity: 0.86

| Variant | Pred Emotion | Pred Intensity | Confidence | Param MAE | Raw Label |
| --- | --- | ---: | ---: | ---: | --- |
| learned | surprise | 0.74 | 0.9797 | 0.025043 | surprise |
| hf_raw | sad | 0.55 | 0.7868 | 0.214026 | sad |
| hf_rules_v4 | surprise | 0.74 | 0.7868 | 0.025043 | sad |

Top parameter differences (`hf_rules_v4`):
- eyeWideLeft: pred=0.6181, target=0.75, abs_diff=0.1319
- browOuterUpRight: pred=0.5555, target=0.68, abs_diff=0.1245
- browOuterUpLeft: pred=0.558, target=0.67, abs_diff=0.112
- eyeWideRight: pred=0.6389, target=0.74, abs_diff=0.1011
- jawOpen: pred=0.4185, target=0.5, abs_diff=0.0815

## Case 9: disgust

- text: 这也太让人倒胃口了。#11
- true_emotion: disgust
- true_intensity: 0.32

| Variant | Pred Emotion | Pred Intensity | Confidence | Param MAE | Raw Label |
| --- | --- | ---: | ---: | ---: | --- |
| learned | disgust | 0.66 | 0.8876 | 0.041123 | disgust |
| hf_raw | disgust | 0.66 | 0.9105 | 0.041123 | disgust |
| hf_rules_v4 | disgust | 0.68 | 0.9105 | 0.043754 | disgust |

Top parameter differences (`hf_rules_v4`):
- eyeSquintLeft: pred=0.4227, target=0.25, abs_diff=0.1727
- mouthFrownRight: pred=0.3397, target=0.18, abs_diff=0.1597
- eyeSquintRight: pred=0.4087, target=0.25, abs_diff=0.1587
- browDownLeft: pred=0.3876, target=0.23, abs_diff=0.1576
- browDownRight: pred=0.3953, target=0.26, abs_diff=0.1353

## Case 10: disgust

- text: 别说了，我不想听细节。
- true_emotion: disgust
- true_intensity: 0.71

| Variant | Pred Emotion | Pred Intensity | Confidence | Param MAE | Raw Label |
| --- | --- | ---: | ---: | ---: | --- |
| learned | angry | 0.72 | 0.9614 | 0.116645 | angry |
| hf_raw | calm | 0.42 | 0.9575 | 0.121238 | calm |
| hf_rules_v4 | disgust | 0.67 | 0.9575 | 0.012641 | calm |

Top parameter differences (`hf_rules_v4`):
- browDownLeft: pred=0.3552, target=0.41, abs_diff=0.0548
- eyeSquintRight: pred=0.3944, target=0.44, abs_diff=0.0456
- mouthFrownLeft: pred=0.3073, target=0.35, abs_diff=0.0427
- browDownRight: pred=0.3811, target=0.42, abs_diff=0.0389
- eyeSquintLeft: pred=0.3929, target=0.43, abs_diff=0.0371

## Case 11: concern

- text: 我还是想再确认一下。#11
- true_emotion: concern
- true_intensity: 0.32

| Variant | Pred Emotion | Pred Intensity | Confidence | Param MAE | Raw Label |
| --- | --- | ---: | ---: | ---: | --- |
| learned | concern | 0.56 | 0.8013 | 0.020044 | concern |
| hf_raw | calm | 0.42 | 0.5536 | 0.029455 | calm |
| hf_rules_v4 | concern | 0.49 | 0.5536 | 0.013307 | calm |

Top parameter differences (`hf_rules_v4`):
- browInnerUp: pred=0.309, target=0.19, abs_diff=0.119
- eyeWideRight: pred=0.1725, target=0.1, abs_diff=0.0725
- eyeWideLeft: pred=0.1579, target=0.11, abs_diff=0.0479
- mouthPressLeft: pred=0.1314, target=0.09, abs_diff=0.0414
- mouthPressRight: pred=0.112, target=0.1, abs_diff=0.012

## Case 12: concern

- text: 看着外面天都黑了，他怎么还没到家。
- true_emotion: concern
- true_intensity: 0.74

| Variant | Pred Emotion | Pred Intensity | Confidence | Param MAE | Raw Label |
| --- | --- | ---: | ---: | ---: | --- |
| learned | happy | 0.68 | 0.6036 | 0.119184 | happy |
| hf_raw | calm | 0.42 | 0.9404 | 0.073259 | calm |
| hf_rules_v4 | concern | 0.58 | 0.9404 | 0.022321 | calm |

Top parameter differences (`hf_rules_v4`):
- browInnerUp: pred=0.3319, target=0.46, abs_diff=0.1281
- eyeWideLeft: pred=0.193, target=0.29, abs_diff=0.097
- eyeWideRight: pred=0.2041, target=0.3, abs_diff=0.0959
- mouthPressRight: pred=0.1549, target=0.24, abs_diff=0.0851
- mouthPressLeft: pred=0.145, target=0.23, abs_diff=0.085

## Case 13: bored

- text: 这些内容听得我直走神。#2
- true_emotion: bored
- true_intensity: 0.39

| Variant | Pred Emotion | Pred Intensity | Confidence | Param MAE | Raw Label |
| --- | --- | ---: | ---: | ---: | --- |
| learned | bored | 0.48 | 0.9474 | 0.003416 | bored |
| hf_raw | disgust | 0.62 | 0.5813 | 0.091580 | disgust |
| hf_rules_v4 | bored | 0.51 | 0.5813 | 0.004010 | disgust |

Top parameter differences (`hf_rules_v4`):
- eyeLookDown: pred=0.2216, target=0.2, abs_diff=0.0216
- browDownRight: pred=0.1513, target=0.13, abs_diff=0.0213
- mouthFrownLeft: pred=0.0964, target=0.08, abs_diff=0.0164
- mouthFrownRight: pred=0.1028, target=0.09, abs_diff=0.0128
- browDownLeft: pred=0.1415, target=0.15, abs_diff=0.0085

## Case 14: bored

- text: 我好像对什么都提不起兴趣。#19
- true_emotion: bored
- true_intensity: 0.85

| Variant | Pred Emotion | Pred Intensity | Confidence | Param MAE | Raw Label |
| --- | --- | ---: | ---: | ---: | --- |
| learned | bored | 0.52 | 0.9027 | 0.015587 | bored |
| hf_raw | calm | 0.42 | 0.9279 | 0.065377 | calm |
| hf_rules_v4 | bored | 0.63 | 0.9279 | 0.007696 | calm |

Top parameter differences (`hf_rules_v4`):
- browDownRight: pred=0.1993, target=0.26, abs_diff=0.0607
- browDownLeft: pred=0.2117, target=0.25, abs_diff=0.0383
- mouthFrownLeft: pred=0.1264, target=0.16, abs_diff=0.0336
- eyeLookLeft: pred=0.2002, target=0.22, abs_diff=0.0198
- mouthFrownRight: pred=0.1403, target=0.15, abs_diff=0.0097

## Case 15: calm

- text: 情绪总算落回来了。#11
- true_emotion: calm
- true_intensity: 0.32

| Variant | Pred Emotion | Pred Intensity | Confidence | Param MAE | Raw Label |
| --- | --- | ---: | ---: | ---: | --- |
| learned | happy | 0.68 | 0.4507 | 0.056799 | happy |
| hf_raw | sad | 0.55 | 0.9002 | 0.059967 | sad |
| hf_rules_v4 | calm | 0.46 | 0.9002 | 0.002701 | sad |

Top parameter differences (`hf_rules_v4`):
- mouthSmileLeft: pred=0.0587, target=0.02, abs_diff=0.0387
- mouthSmileRight: pred=0.0536, target=0.04, abs_diff=0.0136
- browInnerUp: pred=0.0071, target=0.0, abs_diff=0.0071
- browOuterUpLeft: pred=0.0, target=0.0, abs_diff=0.0
- browOuterUpRight: pred=0.0, target=0.0, abs_diff=0.0

## Case 16: calm

- text: 这样安安稳稳的感觉也不错。#7
- true_emotion: calm
- true_intensity: 0.72

| Variant | Pred Emotion | Pred Intensity | Confidence | Param MAE | Raw Label |
| --- | --- | ---: | ---: | ---: | --- |
| learned | surprise | 0.74 | 0.9754 | 0.123007 | surprise |
| hf_raw | calm | 0.42 | 0.8893 | 0.002870 | calm |
| hf_rules_v4 | calm | 0.55 | 0.8893 | 0.001669 | calm |

Top parameter differences (`hf_rules_v4`):
- browInnerUp: pred=0.0182, target=0.0, abs_diff=0.0182
- mouthSmileRight: pred=0.0694, target=0.08, abs_diff=0.0106
- mouthSmileLeft: pred=0.0521, target=0.06, abs_diff=0.0079
- browOuterUpLeft: pred=0.0, target=0.0, abs_diff=0.0
- browOuterUpRight: pred=0.0, target=0.0, abs_diff=0.0

