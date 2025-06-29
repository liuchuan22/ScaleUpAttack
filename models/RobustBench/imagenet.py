from robustbench.utils import load_model
from robustbench.model_zoo.imagenet import normalize_model

def Amini2024MeanSparse(pretrained=True):
    return load_model(model_name="Amini2024MeanSparse", dataset="imagenet", threat_model="Linf")

def Liu2023Comprehensive_Swin_L(pretrained=True):
    return load_model(model_name="Liu2023Comprehensive_Swin-L", dataset="imagenet", threat_model="Linf")

def Bai2024MixedNUTS(pretrained=True):
    return load_model(model_name="Bai2024MixedNUTS", dataset="imagenet", threat_model="Linf")

def Liu2023Comprehensive_ConvNeXt_L(pretrained=True):
    return load_model(model_name="Liu2023Comprehensive_ConvNeXt-L", dataset="imagenet", threat_model="Linf")

def Singh2023Revisiting_ConvNeXt_L(pretrained=True):
    return load_model(model_name="Singh2023Revisiting_ConvNeXt-L-ConvStem", dataset="imagenet", threat_model="Linf")

def Singh2023Revisiting_ViT_B(pretrained=True):
    return load_model(model_name="Singh2023Revisiting_ViT-B-ConvStem", dataset="imagenet", threat_model="Linf")

def Singh2023Revisiting_ViT_S(pretrained=True):
    return load_model(model_name="Singh2023Revisiting_ViT-S-ConvStem", dataset="imagenet", threat_model="Linf")

def Peng2023Robust(pretrained=True):
    return load_model(model_name="Peng2023Robust", dataset="imagenet", threat_model="Linf")

def Chen2024Data_WRN_50(pretrained=True):
    return load_model(model_name="Chen2024Data_WRN_50_2", dataset="imagenet", threat_model="Linf")

def Mo2022When_Swin_B(pretrained=True):
    return load_model(model_name="Mo2022When_Swin-B", dataset="imagenet", threat_model="Linf")

def Mo2022When_ViT_B(pretrained=True):
    return load_model(model_name="Mo2022When_ViT-B", dataset="imagenet", threat_model="Linf")

def Wong2020Fast(pretrained=True):
    return load_model(model_name="Wong2020Fast", dataset="imagenet", threat_model="Linf")


def Engstrom2019Robustness(pretrained=True):
    return load_model(model_name="Engstrom2019Robustness", dataset="imagenet", threat_model="Linf")


def Salman2020Do_R50(pretrained=True):
    return load_model(model_name="Salman2020Do_R50", dataset="imagenet", threat_model="Linf")


def Salman2020Do_R18(pretrained=True):
    return load_model(model_name="Salman2020Do_R18", dataset="imagenet", threat_model="Linf")


def Salman2020Do_50_2(pretrained=True):
    return load_model(model_name="Salman2020Do_50_2", dataset="imagenet", threat_model="Linf")


def Debenedetti2022Light_XCiT_S12(pretrained=True):
    return load_model(model_name="Debenedetti2022Light_XCiT-S12", dataset="imagenet", threat_model="Linf")


def Debenedetti2022Light_XCiT_M12(pretrained=True):
    return load_model(model_name="Debenedetti2022Light_XCiT-M12", dataset="imagenet", threat_model="Linf")


def Debenedetti2022Light_XCiT_L12(pretrained=True):
    return load_model(model_name="Debenedetti2022Light_XCiT-L12", dataset="imagenet", threat_model="Linf")