import os
import torch
import torch.utils.data.dataloader


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_if_exist(path):
    if os.path.exists(path):
        os.remove(path)


def cubes_2_maps(cubes):
    b, c, d, h, w = cubes.shape
    cubes = cubes.permute(0, 2, 1, 3, 4)

    return cubes.contiguous().view(b*d, c, h, w), b, d


def maps_2_cubes(maps, b, d):
    x_b, x_c, x_h, x_w = maps.shape
    maps = maps.contiguous().view(b, d, x_c, x_h, x_w)

    return maps.permute(0, 2, 1, 3, 4)


def inference(model, test_loader, device, metrics_s=None):
    model.eval()
    if metrics_s is not None: metrics_s.reset()

    with torch.no_grad():
        count=0
        for image, depth, depth_scaled, test_indices in test_loader:
            image = image.to(device)

            output = model(image)
            # output = maps_2_cubes(output, batch_size, depth_size)
            depth_new = []
            output_new = []

            for id, index in enumerate(test_indices):
                opti_depth = []
                opti_output = []
                depth_new.append(depth[id, :, index])
                output_new.append(output[id, :, index])
                opti_depth.append((depth[id, :, index-1]/10.0).cuda())
                opti_depth.append((depth[id, :, index]/10.0).cuda())
                opti_output.append(output[id, :, index - 1]/10.0)
                opti_output.append(output[id, :, index]/10.0)

                if metrics_s is not None:
                    metrics_s(torch.stack(output_new, 0).cpu(), torch.stack(depth_new, 0))

        result_s = metrics_s.loss_get()
        print(result_s)


