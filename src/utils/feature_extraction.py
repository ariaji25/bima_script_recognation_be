from directional_feature.DEF import DirectionalElementFeature

DEF = DirectionalElementFeature()


def ExtractFeature(image):
    binary_data = DEF.covertToBinaryPixel(image)
    countour_data = DEF.countourExtraction(binary_data)
    dot_orientation = DEF.dotOrientation(countour_data)
    subareas = DEF.makeSubArea(dot_orientation)
    return DEF.vectorConstraction(subareas)
