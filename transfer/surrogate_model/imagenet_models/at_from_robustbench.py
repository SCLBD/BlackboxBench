from robustbench.utils import load_model

__all__ = ['adv_rawrn_101_2_Peng2023Robust', 'adv_wrn_50_2_Salman2020Do_50_2', 'adv_resnet50_Salman2020Do_R50', 'adv_resnet50_Engstrom2019Robustness',
           'adv_resnet50_Wong2020Fast', 'adv_resnet18_Salman2020Do_R18', 'adv_convnext_l_Liu2023Comprehensive_ConvNeXt_L',
           'adv_convnext_b_Liu2023Comprehensive_ConvNeXt_B', 'adv_convnext_l_convstem_Singh2023Revisiting_ConvNeXt_L_ConvStem',
           'adv_convnext_b_convstem_Singh2023Revisiting_ConvNeXt_B_ConvStem', 'adv_convnext_s_convstem_Singh2023Revisiting_ConvNeXt_S_ConvStem',
           'adv_convnext_t_convstem_Singh2023Revisiting_ConvNeXt_T_ConvStem', 'adv_swin_b_Liu2023Comprehensive_Swin_B',
           'adv_swin_l_Liu2023Comprehensive_Swin_L','adv_xcit_m_Debenedetti2022Light_XCiT_M12',
           'adv_xcit_l_Debenedetti2022Light_XCiT_L12','adv_vit_b_convstem_Singh2023Revisiting_ViT_B_ConvStem',
           'adv_vit_s_convstem_Singh2023Revisiting_ViT_S_ConvStem',]


'=================================================================================================================='
'=================================================Robust Convnet==================================================='
'=================================================================================================================='
def adv_rawrn_101_2_Peng2023Robust(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name="Peng2023Robust", dataset="imagenet", threat_model="Linf")

def adv_wrn_50_2_Salman2020Do_50_2(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name="Salman2020Do_50_2", dataset="imagenet", threat_model="Linf")

def adv_resnet50_Salman2020Do_R50(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name="Salman2020Do_R50", dataset="imagenet", threat_model="Linf")

def adv_resnet50_Engstrom2019Robustness(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name="Engstrom2019Robustness", dataset="imagenet", threat_model="Linf")

def adv_resnet50_Wong2020Fast(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name="Wong2020Fast", dataset="imagenet", threat_model="Linf")

def adv_resnet18_Salman2020Do_R18(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name="Salman2020Do_R18", dataset="imagenet", threat_model="Linf")

def adv_convnext_l_Liu2023Comprehensive_ConvNeXt_L(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name="Liu2023Comprehensive_ConvNeXt-L", dataset="imagenet", threat_model="Linf")

def adv_convnext_b_Liu2023Comprehensive_ConvNeXt_B(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name="Liu2023Comprehensive_ConvNeXt-B", dataset="imagenet", threat_model="Linf")

def adv_convnext_l_convstem_Singh2023Revisiting_ConvNeXt_L_ConvStem(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name="Singh2023Revisiting_ConvNeXt-L-ConvStem", dataset="imagenet", threat_model="Linf")

def adv_convnext_b_convstem_Singh2023Revisiting_ConvNeXt_B_ConvStem(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name="Singh2023Revisiting_ConvNeXt-B-ConvStem", dataset="imagenet", threat_model="Linf")

def adv_convnext_s_convstem_Singh2023Revisiting_ConvNeXt_S_ConvStem(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name="Singh2023Revisiting_ConvNeXt-S-ConvStem", dataset="imagenet", threat_model="Linf")

def adv_convnext_t_convstem_Singh2023Revisiting_ConvNeXt_T_ConvStem(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name="Singh2023Revisiting_ConvNeXt-T-ConvStem", dataset="imagenet", threat_model="Linf")


'=================================================================================================================='
'===============================================Robust MetaFormer=================================================='
'=================================================================================================================='
def adv_swin_b_Liu2023Comprehensive_Swin_B(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name="Liu2023Comprehensive_Swin-B", dataset="imagenet", threat_model="Linf")

def adv_swin_l_Liu2023Comprehensive_Swin_L(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name="Liu2023Comprehensive_Swin-L", dataset="imagenet", threat_model="Linf")

def adv_xcit_m_Debenedetti2022Light_XCiT_M12(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name="Debenedetti2022Light_XCiT-M12", dataset="imagenet", threat_model="Linf")

def adv_xcit_l_Debenedetti2022Light_XCiT_L12(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name="Debenedetti2022Light_XCiT-L12", dataset="imagenet", threat_model="Linf")

def adv_vit_b_convstem_Singh2023Revisiting_ViT_B_ConvStem(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name="Singh2023Revisiting_ViT-B-ConvStem", dataset="imagenet", threat_model="Linf")

def adv_vit_s_convstem_Singh2023Revisiting_ViT_S_ConvStem(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name="Singh2023Revisiting_ViT-S-ConvStem", dataset="imagenet", threat_model="Linf")


## adv models in v2
# def adv_convnext_b(pretrained=True, **kwargs):
#     assert pretrained
#     return load_model(model_name='Liu2023Comprehensive_ConvNeXt-B', dataset='imagenet', threat_model='Linf').model
#
# def adv_resnet50(pretrained=True, **kwargs):
#     assert pretrained
#     return load_model(model_name='Salman2020Do_R50', dataset='imagenet', threat_model='Linf').model
#
# def adv_swin_b(pretrained=True, **kwargs):
#     assert pretrained
#     return load_model(model_name='Liu2023Comprehensive_Swin-B', dataset='imagenet', threat_model='Linf').model
#
# def adv_wrn50(pretrained=True, **kwargs):
#     assert pretrained
#     return load_model(model_name='Salman2020Do_50_2', dataset='imagenet', threat_model='Linf').model
