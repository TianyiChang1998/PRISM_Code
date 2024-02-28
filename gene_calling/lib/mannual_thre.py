import cv2
import numpy as np
import os
from tqdm import tqdm
import re


def relabel_mask(
    intensity,
    plot_column=["X_coor_gaussian", "Y_coor_gaussian"],
    mask=np.array,
    ori_label=int,
    ch_label=int,
    xlim=(-1, 1),
    ylim=(-1, 1.5),
    mode="replace",
    num_per_layer=15,
):
    intensity_tmp = intensity.copy()
    x, y = mask.shape[0], mask.shape[1]

    if mode == "replace":
        data = intensity[intensity['G_layer'] == (ori_label-1)//num_per_layer]
    elif mode == "discard":
        data = intensity[intensity["label"] == ori_label]
        data["label"] = [-1] * len(data)

    data["x"] = (ylim[1] - data[plot_column[1]]) * x / (ylim[1] - ylim[0])
    data["y"] = (data[plot_column[0]] - xlim[0]) * y / (xlim[1] - xlim[0])

    for index, row in tqdm(data.iterrows(), desc=f"relabel{ori_label}"):
        if mask[int(row["x"]), int(row["y"])]:
            data.at[index, "label"] = ch_label

    intensity_tmp.loc[data.index, "label"] = data["label"]
    intensity_tmp = intensity_tmp[intensity_tmp["label"] != -1]

    return intensity_tmp


def relabel(intensity_fra, mask_dir, mode="discard", num_per_layer=15):
    re_label = [
        re.search(r"mask_(\d+)\.png", filename).group(1)
        for filename in os.listdir(mask_dir)
    ]
    if len(re_label) == 0:
        return intensity_fra

    intensity_fra_relabel = intensity_fra.copy()
    for label in re_label:
        label = int(label)
        mask = cv2.imread(
            os.path.join(mask_dir, f"mask_{label}.png"), cv2.IMREAD_GRAYSCALE
        )
        intensity_fra_relabel = relabel_mask(
            intensity_fra_relabel,
            mask=mask,
            ori_label=label,
            ch_label=label,
            mode=mode,
            num_per_layer=num_per_layer,
        )

    return intensity_fra_relabel


def main():
    os.chdir(r"E:\TMC\PRISM_point_typing\dataset\PRISM30_mousebrain")

    # Global variables to store the last point and the mask
    last_point = None
    mask = None

    # Define the mouse callback function
    def draw_mask(event, x, y, flags, param):
        global last_point, mask
        if event == cv2.EVENT_LBUTTONDOWN:
            last_point = (x, y)
            cv2.circle(mask, (x, y), 10, (255, 255, 255), -1)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            cv2.line(mask, last_point, (x, y), (255, 255, 255), 20)
            last_point = (x, y)

    # Display the image
    for layer in range(2):
        window_name = f"layer{layer}, 's' to save, 'c' to continue and 'q' to quit"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, draw_mask)

        img = cv2.imread(f"./figures/layer{layer+1}.jpg")
        mask = np.zeros(img.shape, dtype=np.uint8)

        while True:
            result = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
            cv2.imshow(window_name, result)

            if cv2.waitKey(1) & 0xFF == ord("s"):
                cluster = int(input("cluster you change: "))
                cv2.imwrite(f"./masks/mask_{cluster}.png", mask)
                break
            elif cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "main":
    main()
