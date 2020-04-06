import models

MODEL_DISPATCHER = {
    "base":models.Effinet, # memory 
    'generator':models.Generator,
    'discriminator':models.Discriminators
}

    