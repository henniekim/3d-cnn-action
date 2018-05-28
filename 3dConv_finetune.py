from kerasmodelzoo.models.vgg import vgg16

model = vgg16.model(weights=True, summary=True)
