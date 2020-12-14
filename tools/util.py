
def ship_data_to_cuda(imgs,targets, device):
    f = lambda imgs,targets: ship_data_to_cuda_singe_sample(
        imgs, targets, device=device)
    return tuple(map(list, zip(*map(f, imgs,targets))))


def ship_data_to_cuda_singe_sample(img, target, device):
    img = img.to(device)
    if target is not None:
        target['boxes'] = target['boxes'].to(device)
        target['labels'] = target['labels'].to(device)
        if 'heatmaps' in target:
            target['heatmaps'] = target['heatmaps'].to(device)
    return img, target

def draw_box_in_image(img, box, gt=True, l_w=2):
    import matplotlib.pyplot as plt
    [gx_min, gy_min, gx_max, gy_max] = box.int().cpu().numpy().tolist()
    color = 0 if gt else 1
    img[gy_min:gy_max, gx_min:gx_min + l_w, color] = 255
    img[gy_min:gy_max, gx_max:gx_max + l_w, color] = 255
    img[gy_min:gy_min + l_w, gx_min:gx_max, color] = 255
    img[gy_max:gy_max + l_w, gx_min:gx_max, color] = 255






def Color(val):
    dict={
            'R':'\033[31;1m',
            'G':'\033[32;1m',
            'Y':'\033[33;1m',
            'B':'\033[34;1m',
            'M':'\033[35;1m',
            'C':'\033[36;1m',
            'end':'\033[0m'
            }
    return dict[val]

