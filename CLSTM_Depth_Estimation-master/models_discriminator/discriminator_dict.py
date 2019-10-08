from models_discriminator.resnet_models import resnet18
from models_discriminator.short_resnet_models import short_resnet9
from models_discriminator.customize_models import C_C3D_1, C_C3D_2
from models_discriminator.C2D_models import C_C2D_1


discriminator_dict = {
    'resnet18': resnet18,
    'short_resnet9': short_resnet9,
    'C_C3D_1': C_C3D_1,
    'C_C3D_2': C_C3D_2,
    'C_C2D_1': C_C2D_1
}