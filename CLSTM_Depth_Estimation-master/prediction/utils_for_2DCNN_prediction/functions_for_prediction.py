import os
import torch
import torch.utils.data.dataloader


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_if_exist(path):
    if os.path.exists(path):
        os.remove(path)


def inference(model, test_loader, device, metrics=None):
    model.eval()
    loss = 0
    if metrics is not None: metrics.reset()

    with torch.no_grad():
        for image, depth, depth_scaled in test_loader:
            image = image.to(device)
            output = model(image)

            if metrics is not None:
                metrics(output.cpu(), depth)

        result = metrics.loss_get()
        print(result)

